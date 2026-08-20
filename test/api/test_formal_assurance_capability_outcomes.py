"""FACP-025: Accelerate capability and inference outcome migration.

Acceptance coverage:
- Non-CPU routing requires current capability evidence.
- Simulation remains selectable only in explicit test mode.
- Inference returns observed/delegated evidence, Unknown, Unavailable, or
  Failed — never invented success.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from ipfs_accelerate_py.assurance.capability_outcomes import (
    BUNDLE,
    CLOSED_OUTCOMES,
    EVIDENCE_ID,
    EXPLICIT_TEST_MODE_ENV,
    GOAL_ID,
    INVENTORIED_CAPABILITY_SITES,
    INTERFACE,
    SIMULATION_NAMESPACES,
    TASK_ID,
    UNSAFE_PROMOTION,
    CapabilityOutcome,
    RoutingDisposition,
    SimulationSelection,
    assess_capability_probe,
    begin_inference_attempt,
    bind_inference_observation,
    current_capability_probe,
    is_cpu_backend,
    is_current_capability_evidence,
    is_explicit_test_mode,
    is_non_cpu_backend,
    project_compatibility,
    refuse_compatibility_success,
    resolve_inference_outcome,
    route_backend,
    select_simulation_namespace,
    validate_delegated_inference_receipt,
)

NOW = datetime(2026, 8, 19, 12, 0, tzinfo=timezone.utc)

FORBIDDEN_SUCCESS_MARKERS = frozenset(
    {"success", "ok", "passed", "production_supported", "available", "supported"}
)


def _assert_non_success(result: CapabilityOutcome) -> None:
    assert result.ok is False
    assert result.is_success_disposition is False
    assert result.unsafe_promotion is False
    if result.outcome in {"Observed", "Verified"}:
        assert result.code not in {
            "inference_observed",
            "inference_delegated",
            "effect_observed",
            "verified_admitted",
        }
    compat = result.to_legacy_compat_dict()
    assert compat["status"] != "success"
    assert compat["ok"] is False
    assert compat["disposition"] == "non_success"
    for marker in FORBIDDEN_SUCCESS_MARKERS:
        assert compat.get(marker) is not True


def _live_cuda_probe(**overrides: Any):
    probe = current_capability_probe(
        "cuda",
        probe_identity="probe:cuda:host-a",
        available=True,
        origin="live_observed",
        freshness="current",
        integrity="digest_valid",
        environment="live",
        now=NOW,
    )
    if overrides:
        probe = probe.with_overrides(**overrides)
    return probe


# ---------------------------------------------------------------------------
# Module contract
# ---------------------------------------------------------------------------


def test_module_exports_facp_025_contract() -> None:
    assert TASK_ID == "FACP-025"
    assert GOAL_ID == "FACP-G220"
    assert BUNDLE == "facp/migration/accelerate-outcomes"
    assert EVIDENCE_ID == "facp/accelerate-outcomes@1"
    assert INTERFACE == "AccelerateCapabilityOutcomes@1"
    assert UNSAFE_PROMOTION is False
    assert "Unavailable" in CLOSED_OUTCOMES
    assert "Observed" in CLOSED_OUTCOMES
    assert "Failed" in CLOSED_OUTCOMES
    assert INVENTORIED_CAPABILITY_SITES
    assert any(
        site["seed_id"] == "seed:mock-worker-cuda-true"
        for site in INVENTORIED_CAPABILITY_SITES
    )
    assert "mock_ipfs" in SIMULATION_NAMESPACES
    assert "compat_mock_ipfs" in SIMULATION_NAMESPACES


def test_backend_family_helpers() -> None:
    assert is_cpu_backend("cpu")
    assert is_cpu_backend("cpu:0")
    assert not is_non_cpu_backend("cpu")
    assert is_non_cpu_backend("cuda")
    assert is_non_cpu_backend("cuda:0")
    assert is_non_cpu_backend("openvino")
    assert is_non_cpu_backend("webgpu")


# ---------------------------------------------------------------------------
# Acceptance: Non-CPU routing requires current capability evidence
# ---------------------------------------------------------------------------


def test_cpu_route_admitted_without_non_cpu_probe() -> None:
    decision = route_backend("cpu", probe=None, now=NOW)
    assert decision.admitted is True
    assert decision.backend == "cpu"
    assert decision.disposition is RoutingDisposition.CPU_BASELINE
    assert decision.outcome.ok is True
    assert decision.outcome.details.get("non_cpu_probe_required") is False


def test_cuda_route_without_probe_is_unavailable() -> None:
    decision = route_backend("cuda", probe=None, now=NOW)
    assert decision.admitted is False
    assert decision.backend is None
    assert decision.disposition is RoutingDisposition.REJECTED_MISSING_PROBE
    assert decision.outcome.outcome == "Unavailable"
    assert decision.outcome.code == "capability_probe_missing"
    assert "non_cpu_routing_requires_current_capability_evidence" in decision.reason_codes
    _assert_non_success(decision.outcome)


def test_cuda_route_with_current_live_probe_is_admitted() -> None:
    probe = _live_cuda_probe()
    assert is_current_capability_evidence(probe, now=NOW) is True
    decision = route_backend("cuda:0", probe=probe, now=NOW)
    assert decision.admitted is True
    assert decision.backend == "cuda"
    assert decision.disposition is RoutingDisposition.ADMITTED
    assert decision.outcome.outcome == "Observed"
    assert decision.outcome.ok is True
    assert "probe_identity" in decision.outcome.evidence
    assert "probe_freshness_receipt" in decision.outcome.evidence


def test_simulated_cuda_probe_rejects_non_cpu_routing() -> None:
    probe = _live_cuda_probe(origin="simulated", environment="hermetic")
    assert is_current_capability_evidence(probe, now=NOW) is False
    decision = route_backend("cuda", probe=probe, now=NOW)
    assert decision.admitted is False
    assert decision.disposition is RoutingDisposition.REJECTED_SIMULATED
    assert decision.outcome.outcome == "Simulated"
    assert decision.outcome.details.get("production_routing_forbidden") is True
    _assert_non_success(decision.outcome)


def test_stale_probe_rejects_non_cpu_routing() -> None:
    probe = _live_cuda_probe(freshness="stale")
    decision = route_backend("cuda", probe=probe, now=NOW)
    assert decision.admitted is False
    assert decision.disposition is RoutingDisposition.REJECTED_STALE_PROBE
    assert decision.outcome.outcome == "Rejected"
    assert decision.outcome.code == "capability_probe_stale"
    _assert_non_success(decision.outcome)


def test_expired_probe_receipt_rejects_non_cpu_routing() -> None:
    probe = _live_cuda_probe(
        issued_at=NOW - timedelta(days=2),
        expires_at=NOW - timedelta(seconds=1),
        freshness="current",
    )
    decision = route_backend("cuda", probe=probe, now=NOW)
    assert decision.admitted is False
    assert decision.outcome.code == "capability_probe_stale"
    _assert_non_success(decision.outcome)


def test_declared_probe_origin_cannot_admit_openvino() -> None:
    probe = current_capability_probe(
        "openvino",
        probe_identity="probe:openvino:declared",
        origin="declared",
        freshness="current",
        integrity="structurally_valid",
        environment="hermetic",
        now=NOW,
    )
    assessed = assess_capability_probe(probe, now=NOW)
    assert assessed.outcome == "Rejected"
    assert assessed.code == "capability_probe_weak_origin"
    decision = route_backend("openvino", probe=probe, now=NOW)
    assert decision.admitted is False
    _assert_non_success(decision.outcome)


def test_unavailable_probe_reports_capability_unavailable() -> None:
    probe = _live_cuda_probe(available=False)
    decision = route_backend("cuda", probe=probe, now=NOW)
    assert decision.admitted is False
    assert decision.outcome.outcome == "Unavailable"
    assert decision.outcome.code == "capability_unavailable"
    _assert_non_success(decision.outcome)


def test_probe_backend_mismatch_rejects_route() -> None:
    probe = _live_cuda_probe()
    decision = route_backend("openvino", probe=probe, now=NOW)
    assert decision.admitted is False
    assert decision.outcome.code == "capability_probe_backend_mismatch"
    _assert_non_success(decision.outcome)


# ---------------------------------------------------------------------------
# Acceptance: Simulation selectable only in explicit test mode
# ---------------------------------------------------------------------------


def test_explicit_test_mode_requires_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(EXPLICIT_TEST_MODE_ENV, raising=False)
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "test_formal_assurance_capability_outcomes.py")
    assert is_explicit_test_mode() is False
    assert is_explicit_test_mode(explicit_test_mode=True) is True
    monkeypatch.setenv(EXPLICIT_TEST_MODE_ENV, "1")
    assert is_explicit_test_mode() is True


def test_simulation_namespace_refused_outside_test_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(EXPLICIT_TEST_MODE_ENV, raising=False)
    for namespace in sorted(SIMULATION_NAMESPACES):
        result = select_simulation_namespace(namespace, explicit_test_mode=False)
        assert result.outcome == "Unavailable"
        assert result.code == "simulation_requires_explicit_test_mode"
        assert result.details.get("selection") == (
            SimulationSelection.REFUSED_PRODUCTION.value
        )
        assert result.details.get("compatibility_refusal") is True
        _assert_non_success(result)


def test_simulation_namespace_selectable_in_explicit_test_mode() -> None:
    result = select_simulation_namespace("compat_mock_ipfs", explicit_test_mode=True)
    assert result.outcome == "Simulated"
    assert result.code == "simulation_selected_test_mode"
    assert result.details.get("explicit_test_mode") is True
    assert result.details.get("production_supported") is False
    assert result.details.get("selection") == (
        SimulationSelection.SELECTED_TEST_MODE.value
    )
    # Simulated is never a success disposition.
    _assert_non_success(result)


def test_unknown_simulation_namespace_rejected() -> None:
    result = select_simulation_namespace("not-a-real-namespace", explicit_test_mode=True)
    assert result.outcome == "Rejected"
    assert result.code == "simulation_namespace_unknown"
    _assert_non_success(result)


def test_mock_inference_refused_outside_test_mode() -> None:
    result = resolve_inference_outcome(
        backend="cpu",
        model="bert-base-uncased",
        simulated=True,
        mock_handler=True,
        explicit_test_mode=False,
    )
    assert result.outcome == "Unavailable"
    assert result.code == "inference_simulation_refused_outside_test_mode"
    _assert_non_success(result)


def test_mock_inference_simulated_only_in_test_mode() -> None:
    result = resolve_inference_outcome(
        backend="cpu",
        model="bert-base-uncased",
        simulated=True,
        explicit_test_mode=True,
    )
    assert result.outcome == "Simulated"
    assert result.details.get("production_supported") is False
    _assert_non_success(result)


# ---------------------------------------------------------------------------
# Acceptance: Inference returns observed/delegated, Unknown, Unavailable, Failed
# ---------------------------------------------------------------------------


def test_inference_observed_with_independent_evidence() -> None:
    result = resolve_inference_outcome(
        backend="cpu",
        model="t5-small",
        observation_present=True,
        observation_id="obs-infer-1",
        admission_token="admission:infer-1",
    )
    assert result.outcome == "Observed"
    assert result.code == "inference_observed"
    assert result.ok is True
    assert "independent_effect_observation" in result.evidence
    assert result.envelope.effect == "observed"


def test_inference_delegated_receipt_observed() -> None:
    result = resolve_inference_outcome(
        backend="cpu",
        model="clip",
        delegated_receipt={
            "receipt_id": "receipt:delegated-1",
            "independent_effect_observation": True,
            "signed_receipt": True,
            "signature_valid": True,
            "admission_token": "admission:delegated",
            "environment": "live",
        },
    )
    assert result.outcome == "Observed"
    assert result.code == "inference_delegated"
    assert result.ok is True
    assert "delegated_receipt" in result.evidence
    assert result.details.get("delegated") is True


def test_inference_attempted_unobserved_is_unknown_not_success() -> None:
    result = resolve_inference_outcome(
        backend="cpu",
        attempt_evidenced=True,
        observation_present=False,
    )
    assert result.outcome == "Unknown"
    assert result.code == "inference_unobserved"
    assert result.details.get("success_forbidden_without_observation") is True
    _assert_non_success(result)


def test_bind_inference_without_observation_is_unknown() -> None:
    attempt = begin_inference_attempt(backend="cpu", model="gpt2")
    assert attempt.outcome == "Attempted"
    _assert_non_success(attempt)
    result = bind_inference_observation(attempt, observation_present=False)
    assert result.outcome == "Unknown"
    _assert_non_success(result)


def test_inference_missing_backend_is_unavailable() -> None:
    result = resolve_inference_outcome(
        backend="cpu",
        backend_available=False,
        attempt_evidenced=True,
    )
    assert result.outcome == "Unavailable"
    assert result.code == "inference_backend_unavailable"
    _assert_non_success(result)


def test_inference_absent_evidence_is_unavailable() -> None:
    result = resolve_inference_outcome(backend="cpu")
    assert result.outcome == "Unavailable"
    assert result.code == "inference_evidence_absent"
    _assert_non_success(result)


def test_inference_error_is_failed() -> None:
    result = resolve_inference_outcome(
        backend="cpu",
        error="tokenizer load failed",
    )
    assert result.outcome == "Failed"
    assert result.code == "inference_failed"
    _assert_non_success(result)


def test_non_cpu_inference_without_probe_is_unavailable() -> None:
    result = resolve_inference_outcome(
        backend="cuda",
        model="llama",
        observation_present=True,
        observation_id="obs-cuda",
        probe=None,
    )
    assert result.outcome == "Unavailable"
    assert result.code == "capability_probe_missing"
    _assert_non_success(result)


def test_non_cpu_inference_with_current_probe_and_observation() -> None:
    result = resolve_inference_outcome(
        backend="cuda:0",
        model="llama",
        probe=_live_cuda_probe(),
        observation_present=True,
        observation_id="obs-cuda-2",
        admission_token="admission:cuda",
        now=NOW,
    )
    assert result.outcome == "Observed"
    assert result.code == "inference_observed"
    assert result.ok is True
    assert result.backend == "cuda"


def test_delegated_receipt_missing_is_unavailable() -> None:
    result = validate_delegated_inference_receipt({}, backend="cpu")
    assert result.outcome == "Unavailable"
    assert result.code == "delegated_receipt_missing"
    _assert_non_success(result)


def test_delegated_receipt_unobserved_is_unknown() -> None:
    result = validate_delegated_inference_receipt(
        {"receipt_id": "r1", "signed_receipt": True},
        backend="cpu",
    )
    assert result.outcome == "Unknown"
    assert result.code == "delegated_receipt_unobserved"
    _assert_non_success(result)


def test_delegated_receipt_revoked_is_failed() -> None:
    result = validate_delegated_inference_receipt(
        {
            "receipt_id": "r2",
            "revoked": True,
            "independent_effect_observation": True,
            "signed_receipt": True,
        },
        backend="cpu",
    )
    assert result.outcome == "Failed"
    assert result.code == "delegated_receipt_revoked"
    _assert_non_success(result)


def test_inference_never_invents_success_from_legacy_mock_flags() -> None:
    """Skillset / mock-handler style invented success must not become Observed."""
    result = resolve_inference_outcome(
        backend="cpu",
        model="hf_t5",
        simulated=True,
        mock_handler=True,
        # Even with attempt flags, simulation without test mode stays Unavailable.
        attempt_evidenced=True,
        observation_present=False,
        explicit_test_mode=False,
    )
    assert result.outcome in {"Unavailable", "Unknown", "Failed"}
    assert result.outcome != "Observed"
    _assert_non_success(result)


# ---------------------------------------------------------------------------
# Compatibility refusal
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "legacy,expected_outcome",
    [
        ({"success": True}, "Unavailable"),
        ({"status": "success", "available": True}, "Unavailable"),
        (
            {
                "success": True,
                "implementation_type": "REAL",
                "mock": True,
                "backend": "cpu",
            },
            "Unavailable",
        ),
        (
            {
                "status": "success",
                "note": "Create mock components as fallback",
                "create_mock_model": True,
            },
            "Unavailable",
        ),
        (
            {"status": "success", "attempt_evidenced": True},
            "Unknown",
        ),
        (
            {"backend_available": False, "success": True},
            "Unavailable",
        ),
        (
            {"success": False, "status": "error"},
            "Failed",
        ),
    ],
)
def test_compatibility_refusal_preserves_non_success(
    legacy: dict[str, Any], expected_outcome: str
) -> None:
    projected = refuse_compatibility_success(legacy, operation="inference")
    assert projected.outcome == expected_outcome
    _assert_non_success(projected)
    assert projected.details.get("compatibility_refusal") is True
    # Alias stays aligned.
    assert project_compatibility(legacy, operation="inference").outcome == expected_outcome


def test_compatibility_observation_backed_may_be_observed() -> None:
    projected = refuse_compatibility_success(
        {
            "status": "success",
            "durable_effect": True,
            "independent_effect_observation": True,
            "observation_present": True,
            "backend": "cpu",
        },
        operation="inference",
    )
    assert projected.outcome == "Observed"
    assert projected.ok is True
    assert projected.code == "inference_observed"


def test_outcome_dict_carries_evidence_metadata() -> None:
    result = resolve_inference_outcome(
        backend="cpu",
        observation_present=True,
        observation_id="meta-1",
        admission_token="tok",
    )
    payload = result.to_dict()
    assert payload["task_id"] == TASK_ID
    assert payload["evidence_id"] == EVIDENCE_ID
    assert payload["ok"] is True
    assert payload["outcome"] == "Observed"
    assert payload["unsafe_promotion"] is False
