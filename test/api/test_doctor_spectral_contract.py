"""Authored signed TaskIR fixtures; no benchmark/evaluator or provider calls."""
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from ipfs_accelerate_py.agent_supervisor.entrypoints.facade import Supervisor
from ipfs_accelerate_py.agent_supervisor.prompt.prompt_workflow import (
    PromptAcceptanceRecord, PromptGoalGraph, PromptGoalRecord, PromptOutputRecord,
    PromptTaskRecord, PromptValidationRecord,
)
from ipfs_accelerate_py.agent_supervisor.runtime import doctor_spectral_contract as workflow
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository


PUBLIC_INSTRUCTION = '''Complete the implementation in find_dominant_eigenvalue_and_eigenvector in /app/eigen.py.
"Dominant" means the eigenvalue with the largest magnitude.
The input is a 2D square numpy array with real np.float64 entries, up to size 10x10, and it is not necessarily symmetric so the eigen pair may be complex.
Optimize the function such that it consistently runs faster than the reference numpy solution in /app/eval.py, while satisfying np.allclose(A @ eigenvec, eigenval * eigenvec).
We will run multiple tests and take the median time per call.
You may install system-wide python packages or even use other languages, but the entrypoint must be a Python function in /app/eigen.py.
`/app/eval.py` can help you iterate.
'''
PUBLIC_PREIMAGE = '''import numpy as np


def find_dominant_eigenvalue_and_eigenvector(A: np.ndarray):
    """
    Find the dominant eigenvalue and eigenvector of a general real square matrix.

    Args:
        A: Real-valued square matrix (accepts numpy arrays) up to size 10x10,
        dtype np.float64.

    Returns:
        eigenvalue: Dominant eigenvalue (numpy scalar, potentially complex)
        eigenvector: Corresponding eigenvector (np.ndarray, potentially complex)
    """
    # beat this reference solution!
    eigenvalues, eigenvectors = np.linalg.eig(A)
    idx = np.argmax(np.abs(eigenvalues))
    return eigenvalues[idx], eigenvectors[:, idx]
'''


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def spectral_case(tmp_path, request):
    options = getattr(request, "param", {})
    root = tmp_path / "repository"
    root.mkdir()
    instruction = PUBLIC_INSTRUCTION + options.get("instruction_suffix", "")
    source = PUBLIC_PREIMAGE + options.get("source_suffix", "")
    (root / workflow.INSTRUCTION).write_text(instruction)
    (root / "eigen.py").write_text(source)
    output_path = options.get("output_path", "eigen.py")
    if output_path != "eigen.py":
        (root / output_path).write_text("# authored independent output\n")
    for argv in (["init", "-q"], ["add", "."], ["-c", "user.name=Authored spectral fixture",
                 "-c", "user.email=spectral@example.invalid", "commit", "-qm", "authored public input"]):
        subprocess.run(["git", "-C", str(root), *argv], check=True, capture_output=True)
    profile, lifecycle = tmp_path / "profile", tmp_path / "lifecycle"
    Supervisor.init_local(repository=root, consent=True, profile_dir=profile, lifecycle_dir=lifecycle)
    policy = local.content_identity(local.LOCAL_POLICY)
    validation = PromptValidationRecord(validation_key="authored-structural-check",
        argv=(sys.executable, "-B", "-c", "pass"), policy_cid=policy)
    acceptance = PromptAcceptanceRecord(criterion_key="authored-spectral-candidate",
        criterion="Retain original native validation; this authored fixture is not a numerical proof",
        validation_keys=(validation.validation_key,))
    output = PromptOutputRecord(path=output_path, effect=options.get("effect", "modify"),
        media_type=options.get("media_type", "text/x-python"))
    goal = PromptGoalRecord(goal_key="AUTHORED-GOAL", parent_goal_cid="", dependency_goal_cids=(),
        title="Authored spectral candidate", objective="Stage the signed output",
        rationale="Fixture only", scope_paths=(output_path,), acceptance=(acceptance,))
    task = PromptTaskRecord(task_key="AUTHORED-TASK", goal_cid=goal.goal_cid, dependency_task_cids=(),
        objective="Prepare an inert candidate", rationale="Fixture only", scope_paths=(output_path,),
        outputs=(output,), validations=(validation,), acceptance=(acceptance,), evidence_cids=(),
        policy_roots=(policy,), predicted_files=(output_path,))
    roots = {"request_cid": local.content_identity({"authored": "spectral fixture"}),
        "scan_cid": local.content_identity({"sources": local._sources(root, ["eigen.py", workflow.INSTRUCTION])}),
        "program_root": local.content_identity({"tree": local._git(root, "rev-parse", "HEAD^{tree}")})}
    graph = PromptGoalGraph(**roots, policy_roots=(policy,), goals=(goal,), tasks=(task,), evidence=())
    spec = {"task_key": task.task_key, "scope_paths": [output_path], "dependencies": [],
        "outputs": [{"path": output.path, "effect": output.effect, "media_type": output.media_type}],
        "validations": [{name: local._plain(getattr(validation, name)) for name in (
            "validation_key", "argv", "cwd", "expected_exit_codes", "policy_cid")}],
        "acceptance": [{name: local._plain(getattr(acceptance, name)) for name in (
            "criterion_key", "criterion", "evidence_cids", "validation_keys")}]}
    manifest = local.author_local_benchmark_manifest(repository=root, profile_dir=profile,
        lifecycle_dir=lifecycle, task_specs=[spec], planning_roots=roots)
    admission = local.admit_local_benchmark_plan(graph=graph, manifest=manifest)
    with IntentRepository(tmp_path / "authored-intent.duckdb") as intent:
        materialized = local.materialize_local_benchmark_plan(admission=admission, intent=intent)
        yield {"repository": root, "admission": admission, "intent": intent,
               "task_cid": materialized["task_cids"][0], "state": tmp_path / "spectral-workflow"}


def prepare(case, **overrides):
    return workflow.prepare_spectral_kernel_candidate(**{**case, **overrides})


@pytest.fixture
def trusted_reports(monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime import spectral_eigen_kernel as kernel
    from ipfs_accelerate_py.agent_supervisor.runtime.spectral_target_qualification import expected_probe_sha256
    from ipfs_accelerate_py.agent_supervisor.analysis.spectral_kernel_proof import prove_spectral_adapter_lemmas
    digest = sha(kernel.candidate_source(backend=workflow.BACKEND).encode())
    calls = []
    def qualify(interpreter):
        calls.append(("target", str(interpreter)))
        return {"schema": "spectral-target-numeric-qualification@1", "accepted": True,
            "authored_fixture_only": True,
            "source_sha256": digest, "interpreter": str(interpreter), "backend": workflow.BACKEND,
            "executable": str(interpreter.resolve()), "executable_sha256": sha(interpreter.resolve().read_bytes()),
            "python_version": "AUTHORED", "numpy_version": "AUTHORED", "scipy_version": "AUTHORED", "case_count": 22,
            "target_probe_sha256": expected_probe_sha256(), "provider_calls": 0,
            "kernel_proved": False, "timing_qualified": False, "proof_authority": False,
            "publication_authority": False, "completion_authority": False}
    def prove(source_sha256):
        calls.append(("proof", source_sha256))
        return prove_spectral_adapter_lemmas(kernel_source_sha256=source_sha256)
    monkeypatch.setattr(workflow, "_qualify_target", qualify)
    monkeypatch.setattr(workflow, "_prove", prove)
    return {"digest": digest, "calls": calls, "qualify": qualify, "prove": prove}


def test_public_fixture_bytes_match_reviewed_hashes():
    assert sha(PUBLIC_INSTRUCTION.encode()) == workflow.INSTRUCTION_SHA256
    assert sha(PUBLIC_PREIMAGE.encode()) == workflow.PREIMAGE_SHA256


def test_target_not_selected_is_residual_without_host_inference(spectral_case, monkeypatch):
    monkeypatch.setattr(workflow, "_qualify_target", lambda *_: pytest.fail("no target was selected"))
    monkeypatch.setattr(workflow, "_prove", lambda *_: pytest.fail("no target was selected"))
    result = prepare(spectral_case)
    assert result["status"] == "residual" and result["reason_codes"] == ["target_runtime_unqualified"]
    assert result["target_dependency_qualified"] is False and "artifact" not in result
    assert result["provider_calls"] == 0


def test_candidate_is_inert_and_only_allocated_worker_materializes(spectral_case, trusted_reports, tmp_path, monkeypatch):
    from ipfs_accelerate_py import llm_router
    from ipfs_accelerate_py.agent_supervisor.runtime.doctor_contract_candidate_runner import materialize_doctor_contract_candidate
    monkeypatch.setattr(llm_router, "generate_text", lambda *_a, **_k: pytest.fail("model call forbidden"))
    case = spectral_case
    original_task = case["intent"].get_task(case["task_cid"])
    original_head = local._git(case["repository"], "rev-parse", "HEAD")
    result = prepare(case, target_interpreter=Path(sys.executable))
    assert result["status"] == "candidate_ready" and result["route"] == "doctor_contract_candidate"
    assert result["target_dependency_qualified"] is True
    assert all(result[key] is False for key in ("proof_authority", "publication_authority",
        "completion_authority", "kernel_proved", "whole_program_proved", "performance_proved"))
    artifact = Path(result["artifact"])
    assert artifact.stat().st_mode & 0o222 == 0 and sha(artifact.read_bytes()) == result["sha256"]
    payload = json.loads(artifact.read_bytes())
    edit = payload["edits"][0]
    assert payload["permitted_outputs"] == [{"path": "eigen.py", "effect": "modify", "media_type": "text/x-python"}]
    assert edit["before_sha256"] == workflow.PREIMAGE_SHA256 and edit["after_sha256"] == trusted_reports["digest"]
    assert payload["proof_receipt_id"] == result["check"]["check_receipt_id"]
    assert case["intent"].get_task(case["task_cid"]) == original_task
    assert (case["repository"] / "eigen.py").read_text() == PUBLIC_PREIMAGE
    assert local._git(case["repository"], "rev-parse", "HEAD") == original_head
    workspace = tmp_path / "allocated-worker"
    subprocess.run(["git", "-C", str(case["repository"]), "worktree", "add", "--detach",
                    str(workspace), original_head], check=True, capture_output=True)
    observed = materialize_doctor_contract_candidate(artifact=artifact, expected_sha256=result["sha256"],
        task_cid=case["task_cid"], prompt=json.dumps({"objective_id": original_task["task_alias"]}), workspace=workspace)
    assert observed["status"] == "candidate_materialized" and observed["provider_calls"] == 0
    assert (workspace / "eigen.py").read_bytes() == base64.b64decode(edit["after_bytes_base64"], validate=True)
    assert (case["repository"] / "eigen.py").read_text() == PUBLIC_PREIMAGE
    assert case["intent"].get_task(case["task_cid"]) == original_task


@pytest.mark.parametrize("profile", [None, "different-family@1"])
def test_explicit_other_profile_does_not_select_family(spectral_case, trusted_reports, profile):
    result = prepare(spectral_case, profile=profile, target_interpreter=Path(sys.executable))
    assert result["reason_codes"] == ["spectral_profile_not_selected"] and "artifact" not in result
    assert trusted_reports["calls"] == []


@pytest.mark.parametrize("spectral_case", [{"instruction_suffix": "Changed public meaning\n"},
    {"source_suffix": "raise RuntimeError('task code must never be imported')\n"},
    {"output_path": "different.py"}, {"media_type": "application/json"}], indirect=True)
def test_signed_but_unreviewed_source_or_output_is_residual(spectral_case, trusted_reports):
    result = prepare(spectral_case, target_interpreter=Path(sys.executable))
    assert result["status"] == "residual" and "artifact" not in result
    assert trusted_reports["calls"] == [] and result["provider_calls"] == 0


@pytest.mark.parametrize("gate", ["target", "proof"])
def test_unavailable_target_or_unverified_lemma_has_no_fallback(spectral_case, trusted_reports, monkeypatch, gate):
    if gate == "target":
        monkeypatch.setattr(workflow, "_qualify_target", lambda *_: {"accepted": False, "reason_code": "scipy_unavailable"})
    else:
        monkeypatch.setattr(workflow, "_prove", lambda digest: {"all_lemmas_verified": False,
            "kernel_source_sha256": digest, "reason_code": "solver_unknown"})
    result = prepare(spectral_case, target_interpreter=Path(sys.executable))
    assert result["status"] == "residual" and "artifact" not in result
    assert result["provider_calls"] == 0 and result["target_dependency_qualified"] is (gate == "proof")


@pytest.mark.parametrize("field,value", [("source_sha256", "0" * 64), ("interpreter", "/foreign/python"),
    ("case_count", True), ("scipy_version", ""), ("target_probe_sha256", "invalid"),
    ("backend", "numpy"), ("proof_authority", True), ("target_probe_sha256", "1" * 64),
    ("executable_sha256", "0" * 64)])
def test_affirmative_probe_binding_or_authority_tamper_refuses_handoff(spectral_case, trusted_reports, monkeypatch, field, value):
    def changed(interpreter):
        report = trusted_reports["qualify"](interpreter)
        report[field] = value
        return report
    monkeypatch.setattr(workflow, "_qualify_target", changed)
    with pytest.raises(ValueError):
        prepare(spectral_case, target_interpreter=Path(sys.executable))
    assert not list((spectral_case["repository"] / ".runtime/doctor-contract-candidates").glob("*.json"))


@pytest.mark.parametrize("drift", ["source", "task", "proof_binding", "kernel_source"])
def test_drift_during_qualification_refuses_before_publication(spectral_case, trusted_reports, monkeypatch, drift):
    from ipfs_accelerate_py.agent_supervisor.runtime import spectral_eigen_kernel as kernel
    def changed(digest):
        report = trusted_reports["prove"](digest)
        if drift == "source":
            path = spectral_case["repository"] / "eigen.py"
            path.write_text(PUBLIC_PREIMAGE + "# changed during check\n")
        elif drift == "task":
            task = spectral_case["intent"].get_task(spectral_case["task_cid"])
            spectral_case["intent"].cas_task_status(task_cid=spectral_case["task_cid"],
                expected_revision=task["revision"], new_status="in_progress")
        elif drift == "proof_binding":
            report["kernel_source_sha256"] = "0" * 64
        else:
            original = kernel.candidate_source
            monkeypatch.setattr(kernel, "candidate_source", lambda **kwargs: original(**kwargs) + "# changed\n")
        return report
    monkeypatch.setattr(workflow, "_prove", changed)
    with pytest.raises(ValueError):
        prepare(spectral_case, target_interpreter=Path(sys.executable))
    assert not list((spectral_case["repository"] / ".runtime/doctor-contract-candidates").glob("*.json"))


def test_foreign_task_and_internal_state_refused(spectral_case):
    with pytest.raises(ValueError):
        prepare(spectral_case, task_cid="foreign:task")
    with pytest.raises(ValueError):
        prepare(spectral_case, state=spectral_case["repository"] / "private-state")
    assert not (spectral_case["repository"] / "private-state").exists()


@pytest.mark.parametrize("attack", ["schema", "status", "analysis_hash", "empty_checks",
    "missing_check", "duplicate_name", "sat", "non_boolean_verified", "lapack_authority",
    "scope", "assumptions"])
def test_affirmative_proof_needs_exact_scoped_current_unsat_population(spectral_case, trusted_reports, monkeypatch, attack):
    def changed(digest):
        report = trusted_reports["prove"](digest)
        if attack == "schema":
            report["schema"] = "AUTHORED-forged-schema@1"
        elif attack == "status":
            report["status"] = "unknown"
        elif attack == "analysis_hash":
            report["analysis_source_sha256"] = "0" * 64
        elif attack == "empty_checks":
            report["checks"] = []
        elif attack == "missing_check":
            report["checks"].pop()
        elif attack == "duplicate_name":
            report["checks"][-1]["name"] = report["checks"][0]["name"]
        elif attack == "sat":
            report["checks"][0]["verdict"] = "sat"
        elif attack == "non_boolean_verified":
            report["checks"][0]["verified"] = 1
        elif attack == "lapack_authority":
            report["lapack_implementation_proved"] = True
        elif attack == "scope":
            report["scope"] = "whole program proved"
        else:
            report["assumptions"] = []
        return report
    monkeypatch.setattr(workflow, "_prove", changed)
    with pytest.raises(ValueError):
        prepare(spectral_case, target_interpreter=Path(sys.executable))
    assert not list((spectral_case["repository"] / ".runtime/doctor-contract-candidates").glob("*.json"))


def test_actual_target_checks_and_smt_only_prepare_validation_candidate(spectral_case, tmp_path):
    """Real -I interpreter and scoped SMT; no evaluator or speed qualification."""
    from ipfs_accelerate_py.agent_supervisor.runtime.doctor_contract_candidate_runner import materialize_doctor_contract_candidate
    case = spectral_case
    original_task = case["intent"].get_task(case["task_cid"])
    result = prepare(case, target_interpreter=Path(sys.executable))
    assert result["status"] == "candidate_ready", result
    assert result["target_qualification"]["accepted"] is True
    assert result["target_qualification"]["case_count"] == 22
    assert result["target_qualification"]["interpreter"] == sys.executable
    assert result["conditional_algebra"]["all_lemmas_verified"] is True
    assert result["performance_qualified"] is result["performance_proved"] is False
    assert result["kernel_proved"] is result["whole_program_proved"] is False
    workspace = tmp_path / "real-qualification-worktree"
    subprocess.run(["git", "-C", str(case["repository"]), "worktree", "add", "--detach",
        str(workspace), local._git(case["repository"], "rev-parse", "HEAD")], check=True, capture_output=True)
    observed = materialize_doctor_contract_candidate(artifact=Path(result["artifact"]),
        expected_sha256=result["sha256"], task_cid=case["task_cid"],
        prompt=json.dumps({"objective_id": original_task["task_alias"]}), workspace=workspace)
    assert observed["status"] == "candidate_materialized" and observed["provider_calls"] == 0
    assert observed["publication_authority"] is observed["completion_authority"] is False
    assert (case["repository"] / "eigen.py").read_text() == PUBLIC_PREIMAGE
    assert case["intent"].get_task(case["task_cid"]) == original_task
