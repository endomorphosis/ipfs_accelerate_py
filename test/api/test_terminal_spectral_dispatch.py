"""Explicit spectral selection keeps numerical and speed qualification distinct."""
from contextlib import contextmanager
from pathlib import Path
import sys

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_doctor_dispatch as dispatch
from ipfs_accelerate_py.agent_supervisor.runtime.doctor_spectral_contract import PROFILE
from test.api.test_doctor_spectral_contract import spectral_case  # noqa: F401


def _use_fixture_intent(monkeypatch, case):
    @contextmanager
    def intent(*args, **kwargs):
        yield case["intent"]
    monkeypatch.setattr(dispatch, "IntentRepository", intent)


def test_missing_target_retains_residual_without_provider(spectral_case, monkeypatch):
    case = spectral_case
    _use_fixture_intent(monkeypatch, case)
    result = dispatch.prepare_terminal_doctor_dispatch(repository=case["repository"],
        state=case["state"].parent, admission=case["admission"], task_cid=case["task_cid"],
        contract_profile=PROFILE)
    assert result["status"] == "residual" and result["route"] == "model_router"
    assert result["reason_codes"] == ["target_runtime_unqualified"]
    assert result["provider_calls"] == 0


def test_actual_numeric_candidate_does_not_promote_unqualified_speed(spectral_case, monkeypatch):
    case = spectral_case
    _use_fixture_intent(monkeypatch, case)
    result = dispatch.prepare_terminal_doctor_dispatch(repository=case["repository"],
        state=case["state"].parent, admission=case["admission"], task_cid=case["task_cid"],
        contract_profile=PROFILE, target_interpreter=Path(sys.executable))
    if result["contract_workflow"]["status"] == "residual":
        pytest.skip("isolated runtime lacks numerical dependencies or Z3")
    assert result["contract_workflow"]["status"] == "candidate_ready"
    assert result["status"] == "residual" and result["route"] == "model_router"
    assert result["reason_codes"] == ["spectral_performance_unqualified"]
    assert result["spectral_capabilities"]["finite_numerical_checks_qualified"]
    assert not result["spectral_capabilities"]["benchmark_route_promoted"]
    assert "artifact" not in result
    argv = dispatch.implementation_argv(router=Path("/router"), model="model", reasoning="high",
        timeout=30, semantic_repository=case["repository"], doctor=result)
    assert "--model" in argv and "--doctor-contract-artifact" not in argv
