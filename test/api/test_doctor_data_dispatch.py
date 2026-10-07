"""Actual dispatcher selection for signed finite data transformations."""
from copy import deepcopy
from pathlib import Path

import pytest

from test.api.test_intent_data_transform import _data_case
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository
from benchmarks.agent_supervisor.container_coding import terminal_doctor_dispatch as dispatch


def _admitted(tmp_path, *, public_profile=False):
    case = _data_case(tmp_path / "fixture", public_profile=public_profile)
    proposed = case["proposed"]
    admission = local.admit_local_benchmark_plan(graph=proposed["graph"], manifest=case["manifest"],
        requirement_bindings=proposed["requirement_bindings"])
    with IntentRepository(tmp_path / "intent.duckdb") as intent:
        local.materialize_local_benchmark_plan(admission=admission, intent=intent)
    return case, dict(repository=case["repository"], state=tmp_path, admission=admission,
                      task_cid=proposed["graph"].tasks[0].task_cid)


@pytest.mark.parametrize("public_profile", [False, True])
def test_signed_data_dispatch_selects_existing_candidate_worker(tmp_path, monkeypatch, public_profile):
    case, request = _admitted(tmp_path, public_profile=public_profile)
    monkeypatch.setattr(dispatch, "_installed_provers", lambda: pytest.fail("finite checker discovers no prover"))
    monkeypatch.setattr(dispatch, "prepare_doctor_task_repair", lambda **_: pytest.fail("Python-only repair route selected"))
    result = dispatch.prepare_terminal_doctor_dispatch(**request)
    assert result["status"] == "candidate_ready" and result["route"] == "doctor_contract_candidate"
    assert result["provider_calls"] == 0
    assert not (case["repository"] / "output.jsonl").exists()
    capability = result["symbolic_capabilities"]
    assert capability["contracts"]["selected_profile"] == "finite-ndjson-field-projection@1"
    assert capability["operators"]["selected_workflow"] == "reviewed_finite_record_projection"
    assert capability["finite_data_check"]["reported"] is True
    assert capability["finite_data_check"]["kernel_proof"] is False
    assert capability["proof"]["local_contract_proof_reported"] is False
    assert capability["contracts"]["whole_task_behavior_verified"] is False
    assert capability["completion_authority"] is capability["proof_authority"] is False
    assert "local_contract_proof_not_reported" in capability["gap_codes"]
    assert "local_prover_unavailable" not in capability["gap_codes"]
    assert capability["contracts"]["named_structural_check"] is public_profile
    argv = dispatch.implementation_argv(router=Path("/router"), model="unused", reasoning="high",
        timeout=30, semantic_repository=case["repository"], doctor=result)
    assert "--doctor-contract-artifact" in argv and "--model" not in argv


def test_unsupported_data_retains_explicit_residual(tmp_path, monkeypatch):
    from ipfs_datasets_py.logic.software_contracts import finite_record_projection as shared
    _, request = _admitted(tmp_path)
    def unsupported(*_):
        raise shared.FiniteRecordProjectionError("unsupported")
    monkeypatch.setattr(shared, "synthesize_finite_record_projection", unsupported)
    result = dispatch.prepare_terminal_doctor_dispatch(**request)
    assert result["status"] == "residual" and result["route"] == "model_router"
    assert result["reason_codes"] == ["finite_data_source_outside_reviewed_profile"]
    assert "artifact" not in result
    assert result["contract_workflow"]["contract_index"]["finite_check_recorded"] is False
    assert result["symbolic_capabilities"]["finite_data_check"]["reported"] is False


def test_unsigned_selector_or_conflicting_profile_cannot_select_data_route(tmp_path):
    _, request = _admitted(tmp_path)
    with pytest.raises(ValueError, match="conflicts"):
        dispatch.prepare_terminal_doctor_dispatch(**request, contract_profile="wsgi-header-controls@1")
    tampered = deepcopy(request)
    tampered["admission"]["manifest"]["payload"]["intent_requirements"]["contract_json"] += " "
    with pytest.raises(ValueError):
        dispatch.prepare_terminal_doctor_dispatch(**tampered)
    assert not (tmp_path / "doctor-data-workflow").exists()


def test_output_parent_must_exist_in_the_allocated_baseline(tmp_path):
    import subprocess
    from ipfs_accelerate_py.agent_supervisor.runtime.doctor_data_contract import _output_parent_available
    case = _data_case(tmp_path / "fixture")
    root = case["repository"]
    (root / "tracked").mkdir()
    (root / "tracked/input.jsonl").write_bytes(case["input_bytes"])
    subprocess.run(["git", "-C", str(root), "add", "tracked/input.jsonl"], check=True)
    subprocess.run(["git", "-C", str(root), "-c", "user.name=Authored data", "-c",
        "user.email=data@example.invalid", "commit", "-qm", "Tracked output parent"], check=True)
    baseline = local._git(root, "rev-parse", "HEAD")
    (root / "empty").mkdir()
    (root / "alias").symlink_to(root / "tracked", target_is_directory=True)
    assert _output_parent_available(root, "output.jsonl", baseline)
    assert _output_parent_available(root, "tracked/output.jsonl", baseline)
    for path in ("absent/output.jsonl", "empty/output.jsonl", "alias/output.jsonl"):
        assert not _output_parent_available(root, path, baseline)
