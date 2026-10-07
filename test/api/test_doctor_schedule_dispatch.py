"""Reviewed finite scheduling reaches the candidate route without proof inflation."""
from copy import deepcopy
from pathlib import Path

import pytest

from test.api.test_intent_interval_schedule import _schedule_case
from test.api.test_doctor_schedule_contract import isolated_schedule_resources  # noqa: F401
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository
from benchmarks.agent_supervisor.container_coding import terminal_doctor_dispatch as dispatch


def _admitted(tmp_path):
    case = _schedule_case(tmp_path / "fixture")
    proposed = case["proposed"]
    admission = local.admit_local_benchmark_plan(graph=proposed["graph"], manifest=case["manifest"],
        requirement_bindings=proposed["requirement_bindings"])
    with IntentRepository(tmp_path / "intent.duckdb") as intent:
        local.materialize_local_benchmark_plan(admission=admission, intent=intent)
    return case, dict(repository=case["repository"], state=tmp_path, admission=admission,
                      task_cid=proposed["graph"].tasks[0].task_cid)


def test_signed_schedule_selects_checked_candidate_route(tmp_path, monkeypatch):
    case, request = _admitted(tmp_path)
    monkeypatch.setattr(dispatch, "_installed_provers", lambda: pytest.fail("finite scheduling discovers no kernel prover"))
    monkeypatch.setattr(dispatch, "prepare_doctor_task_repair", lambda **_: pytest.fail("Python-only repair route selected"))
    result = dispatch.prepare_terminal_doctor_dispatch(**request)
    assert result["status"] == "candidate_ready" and result["route"] == "doctor_contract_candidate"
    assert result["provider_calls"] == 0
    assert not (case["repository"] / "output.json").exists()
    capability = result["symbolic_capabilities"]
    assert capability["contracts"]["selected_profile"] == "finite-integer-interval-schedule@1"
    assert capability["operators"]["selected_workflow"] == "reviewed_finite_interval_schedule"
    checked = capability["finite_schedule_check"]
    assert checked["reported"] is True and checked["kernel_proof"] is False
    assert checked["optimality_verified"] is False
    assert checked["scope"] == "exact_signed_finite_schedule_feasibility"
    assert checked["solver_status"] == "sat"
    assert capability["proof"]["local_contract_proof_reported"] is False
    assert capability["contracts"]["whole_task_behavior_verified"] is False
    assert capability["contracts"]["named_structural_check"] is False
    assert capability["completion_authority"] is capability["proof_authority"] is False
    assert "local_contract_proof_not_reported" in capability["gap_codes"]
    assert "local_prover_unavailable" not in capability["gap_codes"]
    argv = dispatch.implementation_argv(router=Path("/router"), model="unused", reasoning="high",
        timeout=30, semantic_repository=case["repository"], doctor=result)
    assert "--doctor-contract-artifact" in argv and "--model" not in argv


@pytest.mark.parametrize("status", ["unknown", "timeout", "unavailable", "cancelled", "error"])
def test_inconclusive_schedule_dispatch_retains_explicit_model_residual(tmp_path, monkeypatch, status):
    from ipfs_datasets_py.logic.software_contracts import finite_interval_schedule as shared
    case, request = _admitted(tmp_path)
    solve = shared.solve_finite_interval_schedule

    def inconclusive(*args, **kwargs):
        result = solve(*args, **kwargs)
        result.pop("output_bytes", None)
        result["status"] = status
        return result

    monkeypatch.setattr(shared, "solve_finite_interval_schedule", inconclusive)
    result = dispatch.prepare_terminal_doctor_dispatch(**request)
    assert result["status"] == "residual" and result["route"] == "model_router"
    assert result["reason_codes"] == ["finite_schedule_solver_" + status]
    assert result["provider_calls"] == 0
    assert "artifact" not in result
    assert result["contract_workflow"]["contract_index"]["finite_check_recorded"] is False
    assert result["symbolic_capabilities"]["finite_schedule_check"]["reported"] is False
    assert result["symbolic_capabilities"]["finite_schedule_check"]["solver_status"] == status
    assert not (case["repository"] / "output.json").exists()


def test_unsigned_selector_or_conflicting_profile_cannot_select_schedule_route(tmp_path):
    _, request = _admitted(tmp_path)
    with pytest.raises(ValueError, match="conflicts"):
        dispatch.prepare_terminal_doctor_dispatch(**request, contract_profile="wsgi-header-controls@1")
    tampered = deepcopy(request)
    tampered["admission"]["manifest"]["payload"]["intent_requirements"]["contract_json"] += " "
    with pytest.raises(ValueError):
        dispatch.prepare_terminal_doctor_dispatch(**tampered)
    assert not (tmp_path / "doctor-schedule-workflow").exists()
