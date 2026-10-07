"""Signed finite scheduling witnesses remain checked proposals until native replay.

These tests exercise an explicitly reviewed integer model. A feasible witness is
neither a kernel proof nor an interpretation/optimality/completion certificate.
"""
from copy import deepcopy
import base64
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from test.api.test_intent_interval_schedule import _schedule_case
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository


@pytest.fixture(autouse=True)
def isolated_schedule_resources(tmp_path, monkeypatch):
    """Own a real bounded scheduler ledger instead of an account-wide facade."""
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer import resource_scheduler as resources
    monkeypatch.setenv(resources.DEFAULT_STATE_ENV, str(tmp_path / "proof-resource-pool.json"))
    monkeypatch.setenv(resources.DEFAULT_PROOF_PROFILE_ENV, resources.LOCAL_BENCHMARK_PROOF_PROFILE)
    monkeypatch.setenv(resources.DEFAULT_PROOF_RECOVERY_ENV, "1")
    monkeypatch.setenv("IPFS_DATASETS_PROOF_RESOURCE_SAFETY", "1")
    yield
    snapshot = resources.get_global_resource_scheduler().snapshot()
    assert snapshot["active_lease_count"] == snapshot["waiting_request_count"] == 0


@pytest.fixture
def schedule_case(tmp_path):
    case = _schedule_case(tmp_path / "fixture")
    proposed = case["proposed"]
    case["admission"] = local.admit_local_benchmark_plan(graph=proposed["graph"], manifest=case["manifest"],
        requirement_bindings=proposed["requirement_bindings"])
    with IntentRepository(tmp_path / "intent.duckdb") as intent:
        materialized = local.materialize_local_benchmark_plan(admission=case["admission"], intent=intent)
        yield {**case, "intent": intent, "task_cid": materialized["task_cids"][0], "state": tmp_path / "doctor"}


def _prepare(case):
    from ipfs_accelerate_py.agent_supervisor.runtime.doctor_data_contract import prepare_interval_schedule_candidate
    return prepare_interval_schedule_candidate(**{name: case[name] for name in (
        "repository", "admission", "intent", "task_cid", "state")})


def _assert_no_publication(case):
    assert not (case["repository"] / "output.json").exists()
    assert local._git(case["repository"], "rev-parse", "HEAD") == case["baseline_commit"]
    assert not list((case["repository"] / ".runtime/doctor-contract-candidates").glob("*.json"))


def test_checked_schedule_is_inert_until_allocated_materialization(schedule_case, tmp_path, monkeypatch):
    from ipfs_accelerate_py import llm_router
    from ipfs_accelerate_py.agent_supervisor.runtime.doctor_contract_candidate_runner import materialize_doctor_contract_candidate
    from ipfs_datasets_py.logic.software_contracts.finite_interval_schedule import (
        FiniteIntervalScheduleContract, check_finite_interval_schedule, verify_finite_schedule_check,
    )

    def forbidden(*args, **kwargs):
        pytest.fail("finite scheduling must not call a model")

    for name in ("generate_text", "generate_text_batch", "get_llm_provider"):
        monkeypatch.setattr(llm_router, name, forbidden)
    case = schedule_case
    before = case["intent"].get_task(case["task_cid"])
    result = _prepare(case)
    assert result["status"] == "candidate_ready", result
    assert result["route"] == "doctor_contract_candidate"
    assert result["operator"] == "finite-integer-interval-schedule@1"
    assert result["provider_calls"] == 0
    assert result["publication_authority"] is result["completion_authority"] is False
    check = result["check"]["check"]
    assert check["status"] == "checked" and check["evidence_kind"] == "finite_schedule_check"
    assert check["kernel_checked"] is False
    assert result["kernel_proved"] is result["whole_program_proved"] is result["proof_authority"] is False
    assert result["contract_index"]["hydrated"] is True
    assert result["contract_index"]["finite_check_recorded"] is True
    assert result["contract_index"]["active_receipt_ids"] == []
    assert case["intent"].get_task(case["task_cid"]) == before
    assert not (case["repository"] / "output.json").exists()
    assert local._git(case["repository"], "rev-parse", "HEAD") == case["baseline_commit"]
    artifact = Path(result["artifact"])
    assert artifact.stat().st_mode & 0o222 == 0
    raw = artifact.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == result["sha256"]
    payload = json.loads(raw)
    assert payload["manifest_cid"] == local.content_identity(case["manifest"])
    assert payload["task_revision"] == before["revision"]
    assert len(payload["edits"]) == 1
    edit = payload["edits"][0]
    candidate = base64.b64decode(edit["after_bytes_base64"], validate=True)
    witness = json.loads(candidate)
    assert witness["schema"] == "finite-interval-schedule-witness@1"
    assert [row["id"] for row in witness["assignments"]] == [row["id"] for row in case["input_value"]["jobs"]]
    assert edit["before_sha256"] is None and edit["effect"] == "create"
    assert edit["after_sha256"] == hashlib.sha256(candidate).hexdigest()
    assert payload["proof_receipt_id"] == result["check"]["check_receipt_id"]
    assert check["input_sha256"] == hashlib.sha256(case["input_bytes"]).hexdigest()
    assert check["output_sha256"] == edit["after_sha256"]
    contract = FiniteIntervalScheduleContract()
    checked = check_finite_interval_schedule(case["input_bytes"], candidate, contract)
    assert checked["status"] == "checked"
    verify_finite_schedule_check(check, input_bytes=case["input_bytes"], output_bytes=candidate, contract=contract)
    workspace = tmp_path / "allocated"
    subprocess.run(["git", "-C", str(case["repository"]), "worktree", "add", "--detach", str(workspace),
        case["baseline_commit"]], check=True, capture_output=True)
    materialized = materialize_doctor_contract_candidate(artifact=artifact,
        expected_sha256=result["sha256"], task_cid=case["task_cid"],
        prompt=json.dumps({"objective_id": before["task_alias"]}), workspace=workspace)
    assert materialized["status"] == "candidate_materialized"
    assert materialized["writes"] == [{"path": "output.json", "effect": "create",
        "after_sha256": edit["after_sha256"], "write_mode": "exclusive_create"}]
    assert (workspace / "output.json").read_bytes() == candidate
    assert (workspace / "input.json").read_bytes() == case["input_bytes"]
    # Independently authored public validator imports no shared checker.
    assert "finite_interval_schedule" not in case["public_check_bytes"].decode()
    subprocess.run(list(case["proposed"]["graph"].tasks[0].validations[0].argv),
        cwd=workspace, check=True, capture_output=True, timeout=10)
    assert not (case["repository"] / "output.json").exists()
    assert case["intent"].get_task(case["task_cid"]) == before
    assert materialized["publication_authority"] is materialized["completion_authority"] is False


@pytest.mark.parametrize("drift", ["input", "public_check", "created_output", "terminal_task"])
def test_source_and_task_drift_refuse_schedule_candidate(schedule_case, drift):
    case = schedule_case
    if drift == "input":
        (case["repository"] / "input.json").write_bytes(b"{}\n")
    elif drift == "public_check":
        (case["repository"] / "public_check.py").write_text("pass\n")
    elif drift == "created_output":
        (case["repository"] / "output.json").write_bytes(b"{}\n")
    else:
        task = case["intent"].get_task(case["task_cid"])
        case["intent"].cas_task_status(task_cid=case["task_cid"], expected_revision=task["revision"],
            new_status="cancelled")
    before = case["intent"].get_task(case["task_cid"])
    with pytest.raises(ValueError):
        _prepare(case)
    assert case["intent"].get_task(case["task_cid"]) == before
    assert local._git(case["repository"], "rev-parse", "HEAD") == case["baseline_commit"]
    assert not list((case["repository"] / ".runtime/doctor-contract-candidates").glob("*.json"))


def test_unreviewed_intent_retains_schedule_residual(tmp_path):
    from ipfs_accelerate_py.agent_supervisor.planning.intent_symbolic_planning import build_intent_symbolic_plan
    case = _schedule_case(tmp_path / "fixture")
    contract = deepcopy(case["contract"])
    contract["schema"] = "intent-plan-requirement-contract@2"
    del contract["reviewed_interval_schedule"]
    payload = case["manifest"]["payload"]
    manifest = local.author_local_benchmark_manifest(repository=case["repository"], profile_dir=case["profile"],
        lifecycle_dir=case["lifecycle"], task_specs=payload["tasks"], planning_roots=payload["planning_roots"],
        intent_requirements=contract)
    proposed = build_intent_symbolic_plan(contract, manifest=manifest)
    admission = local.admit_local_benchmark_plan(graph=proposed["graph"], manifest=manifest,
        requirement_bindings=proposed["requirement_bindings"])
    with IntentRepository(tmp_path / "intent.duckdb") as intent:
        materialized = local.materialize_local_benchmark_plan(admission=admission, intent=intent)
        case.update(admission=admission, intent=intent, task_cid=materialized["task_cids"][0], state=tmp_path / "doctor")
        before = intent.get_task(case["task_cid"])
        result = _prepare(case)
        assert result["status"] == "residual" and result["provider_calls"] == 0
        assert result["publication_authority"] is result["completion_authority"] is False
        assert "artifact" not in result
        assert intent.get_task(case["task_cid"]) == before
        _assert_no_publication(case)


@pytest.mark.parametrize("attack", ["overlap", "unknown_job", "source_drift", "task_drift", "forged_check"])
def test_independent_schedule_replay_refuses_producer_or_binding_drift(schedule_case, monkeypatch, attack):
    from ipfs_datasets_py.logic.software_contracts import finite_interval_schedule as shared
    case = schedule_case
    solve = shared.solve_finite_interval_schedule
    if attack == "forged_check":
        checker = shared.check_finite_interval_schedule
        observed = []

        def forged(*args, **kwargs):
            receipt = checker(*args, **kwargs)
            if len(observed) == 1:
                receipt["output_sha256"] = "0" * 64
            observed.append(True)
            return receipt

        # The solver also checks its proposal. Corrupt the consumer replay,
        # then let independent receipt verification run normally.
        monkeypatch.setattr(shared, "check_finite_interval_schedule", forged)
    else:
        def changed(*args, **kwargs):
            result = solve(*args, **kwargs)
            assert result["status"] == "sat"
            if attack in {"overlap", "unknown_job"}:
                witness = json.loads(result["output_bytes"])
                if attack == "unknown_job":
                    witness["assignments"][0]["id"] = "forged-job"
                else:
                    for assignment, job in zip(witness["assignments"], case["input_value"]["jobs"]):
                        assignment["start"] = 1
                        assignment["end"] = 1 + job["duration"]
                result["output_bytes"] = json.dumps(witness).encode()
            elif attack == "source_drift":
                (case["repository"] / "input.json").write_bytes(b"{}\n")
            else:
                task = case["intent"].get_task(case["task_cid"])
                case["intent"].cas_task_status(task_cid=case["task_cid"],
                    expected_revision=task["revision"], new_status="in_progress")
            return result

        monkeypatch.setattr(shared, "solve_finite_interval_schedule", changed)
    with pytest.raises(ValueError):
        _prepare(case)
    _assert_no_publication(case)
    assert case["intent"].get_task(case["task_cid"])["status"] in {"ready", "in_progress"}


@pytest.mark.parametrize("field", ["check_receipt_id", "task_cid", "output_path"])
def test_schedule_index_cannot_replace_bound_check(schedule_case, monkeypatch, field):
    from ipfs_accelerate_py.agent_supervisor.runtime import doctor_data_contract as workflow
    case = schedule_case
    persist = workflow._persist_index

    def mutate(**kwargs):
        result = persist(**kwargs)
        kwargs["check"][field] = "untrusted-index-replacement"
        return result

    monkeypatch.setattr(workflow, "_persist_index", mutate)
    before = case["intent"].get_task(case["task_cid"])
    with pytest.raises(ValueError):
        _prepare(case)
    assert case["intent"].get_task(case["task_cid"]) == before
    _assert_no_publication(case)


@pytest.mark.parametrize("status", ["unknown", "timeout"])
def test_indeterminate_solver_result_never_becomes_a_candidate(schedule_case, monkeypatch, status):
    from ipfs_datasets_py.logic.software_contracts import finite_interval_schedule as shared
    solve = shared.solve_finite_interval_schedule

    def indeterminate(*args, **kwargs):
        result = solve(*args, **kwargs)
        result.pop("output_bytes", None)
        result.update(status=status, reason_codes=["injected_solver_" + status])
        return result

    monkeypatch.setattr(shared, "solve_finite_interval_schedule", indeterminate)
    before = schedule_case["intent"].get_task(schedule_case["task_cid"])
    result = _prepare(schedule_case)
    assert result["status"] == "residual" and result["provider_calls"] == 0
    assert result["reason_codes"] and any(status in code for code in result["reason_codes"])
    assert result["contract_index"]["finite_check_recorded"] is False
    assert result["contract_index"]["active_receipt_ids"] == []
    assert "artifact" not in result and "check" not in result
    assert schedule_case["intent"].get_task(schedule_case["task_cid"]) == before
    _assert_no_publication(schedule_case)


def test_real_unsatisfiable_model_stays_residual_without_completion(tmp_path):
    default = _schedule_case(tmp_path / "input-template")
    problem = deepcopy(default["input_value"])
    # Both jobs are individually admissible but cannot share this fixed slot:
    # one consumes 1 unit and the other 2 units of a 2-unit resource.
    for job in problem["jobs"]:
        job.update(release=0, deadline=4, duration=4)
    case = _schedule_case(tmp_path / "fixture", input_value=problem)
    proposed = case["proposed"]
    admission = local.admit_local_benchmark_plan(graph=proposed["graph"], manifest=case["manifest"],
        requirement_bindings=proposed["requirement_bindings"])
    with IntentRepository(tmp_path / "intent.duckdb") as intent:
        materialized = local.materialize_local_benchmark_plan(admission=admission, intent=intent)
        case.update(admission=admission, intent=intent, task_cid=materialized["task_cids"][0], state=tmp_path / "doctor")
        before = intent.get_task(case["task_cid"])
        result = _prepare(case)
        assert result["status"] == "residual" and result["provider_calls"] == 0
        assert any("unsat" in code for code in result["reason_codes"])
        assert result["contract_index"]["finite_check_recorded"] is False
        assert result["contract_index"]["active_receipt_ids"] == []
        assert "artifact" not in result and "check" not in result
        assert intent.get_task(case["task_cid"]) == before
        _assert_no_publication(case)


@pytest.mark.parametrize("producer_result", [
    None, {"status": "optimal"}, {"status": True}, {"status": "sat"},
    {"status": "sat", "output_bytes": "not exact bytes"},
    {"status": "unknown", "output_bytes": b"{}"},
    {"status": "unsat", "output_bytes": b"{}"},
])
def test_malformed_solver_results_are_hard_refusals(schedule_case, monkeypatch, producer_result):
    from ipfs_datasets_py.logic.software_contracts import finite_interval_schedule as shared
    monkeypatch.setattr(shared, "solve_finite_interval_schedule", lambda *_: producer_result)
    before = schedule_case["intent"].get_task(schedule_case["task_cid"])
    with pytest.raises(ValueError):
        _prepare(schedule_case)
    assert schedule_case["intent"].get_task(schedule_case["task_cid"]) == before
    _assert_no_publication(schedule_case)
