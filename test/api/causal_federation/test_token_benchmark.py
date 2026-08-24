"""Closed unavailable and dormant-boundary contracts for CASF-041."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
RUNNER_RELATIVE_PATH = "benchmarks/agent_supervisor/causal_event_federation/run_token.py"
MANIFEST_RELATIVE_PATH = "benchmarks/agent_supervisor/causal_event_federation/token_manifest.json"
TEST_RELATIVE_PATH = "test/api/causal_federation/test_token_benchmark.py"
MATRIX_RELATIVE_PATH = "benchmarks/agent_supervisor/causal_event_federation/matrix.yaml"
RUNNER_PATH = ROOT / RUNNER_RELATIVE_PATH
MANIFEST_PATH = ROOT / MANIFEST_RELATIVE_PATH
MATRIX_PATH = ROOT / MATRIX_RELATIVE_PATH
SPEC = importlib.util.spec_from_file_location("casf_token_benchmark", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
token = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(token)

OWNED_PATHS = (MANIFEST_RELATIVE_PATH, RUNNER_RELATIVE_PATH, TEST_RELATIVE_PATH)
PreparedResult = tuple[dict[str, Any], Any, Path, dict[str, str]]


def _run(*args: str, cwd: Path) -> str:
    return subprocess.run(list(args), cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def _prepared_clone(path: Path) -> tuple[Path, Any]:
    repository = path / "repository"
    _run("git", "clone", "-q", "--shared", str(ROOT), str(repository), cwd=path)
    for relative_path in OWNED_PATHS:
        destination = repository / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative_path, destination)
    _run("git", "add", "-f", *OWNED_PATHS, cwd=repository)
    _run("git", "-c", "user.name=CASF Test", "-c", "user.email=casf-test@example.invalid", "commit", "-q", "--allow-empty", "-m", "CASF-041 fixture", cwd=repository)
    spec = importlib.util.spec_from_file_location(f"casf_token_{path.name}", repository / RUNNER_RELATIVE_PATH)
    assert spec is not None and spec.loader is not None
    cloned = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cloned)
    return repository, cloned


@pytest.fixture(scope="module")
def prepared_result(tmp_path_factory: pytest.TempPathFactory) -> PreparedResult:
    repository, cloned = _prepared_clone(tmp_path_factory.mktemp("token-unavailable"))
    result = cloned.run_benchmark(repository=repository, identities=None)
    return result, cloned, repository, cloned.repository_identity(repository)


def _rehash(module: Any, result: dict[str, Any]) -> dict[str, Any]:
    result["content_sha256"] = module.result_content_sha256(result)
    return result


def _valid_admission(module: Any) -> Any:
    return module.TokenAdmission(
        schema=module.LIVE_ATTESTATION_SCHEMA,
        baseline_supervisor_processes=1,
        candidate_supervisor_processes=12,
        state_authority="authenticated_typed_quack",
        quack_receipt_ref=f"sha256:{'a' * 64}",
        direct_database_access_permitted=False,
        ducklake_scheduling_authority_permitted=False,
    )


def _valid_observation(module: Any) -> dict[str, Any]:
    return {
        "schema": module.LIVE_OBSERVATION_SCHEMA,
        "comparison_identity": "same-host-tasks-providers-tests-proofs-budgets",
        "baseline_supervisor_processes": 1,
        "candidate_supervisor_processes": 12,
        "state_transport": "authenticated_typed_quack",
        "quack_receipt_ref": f"sha256:{'a' * 64}",
        "direct_database_access_used": False,
        "ducklake_scheduling_authority_used": False,
        "zero_tolerance_gate_failures": {gate: 0 for gate in module.ZERO_TOLERANCE_GATES},
        "repeated_context_token_reduction_percent": 50,
        "input_tokens_per_accepted_criterion_reduction_percent": 40,
        "duplicate_model_call_reduction_percent": 60,
        "eligible_semantic_capsule_reuse_percent": 70,
        "complete_board_scan_reduction_percent": 80,
    }


def test_manifest_freezes_cross_supervisor_profile_matrix_gates_and_unavailability() -> None:
    manifest = token.load_manifest(MANIFEST_PATH)
    assert token.validate_matrix_binding(manifest, MATRIX_PATH) == manifest["matrix_binding"]
    assert manifest["state"] == "capability_unavailable"
    assert manifest["authoritative"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["execution"] == token._EXECUTION
    assert manifest["capacity_preflight"] == token.capacity_preflight()
    assert manifest["comparison"] == token._COMPARISON
    assert manifest["token_gates"] == {
        "minimum_repeated_context_token_reduction_percent": 50,
        "minimum_input_tokens_per_accepted_criterion_reduction_percent": 40,
        "minimum_duplicate_model_call_reduction_percent": 60,
        "minimum_eligible_semantic_capsule_reuse_percent": 70,
        "minimum_complete_board_scan_reduction_percent": 80,
    }
    assert tuple(manifest["zero_tolerance_gates"]) == token.ZERO_TOLERANCE_GATES
    assert tuple(manifest["future_identity_requirements"]) == token.REQUIRED_IDENTITIES
    assert tuple(manifest["nonclaims"]) == token.NONCLAIMS


def test_unavailable_result_is_closed_content_addressed_tree_and_source_bound(prepared_result: PreparedResult) -> None:
    result, cloned, repository, identity = prepared_result
    assert cloned.validate_result(result, repository=repository, current_identity=identity) == result
    assert result["schema"] == cloned.RESULT_SCHEMA
    assert result["availability"] == "unavailable"
    assert result["execution_status"] == "not_run"
    assert result["ran"] is False
    assert result["qualified"] is False
    assert result["metrics_omitted"] is True
    assert result["authoritative"] is False
    assert result["promotion_eligible"] is False
    assert result["repository_binding"] == {**identity, "clean": True, "observed_before_and_after": True}
    manifest_raw, runner_raw = (repository / MANIFEST_RELATIVE_PATH).read_bytes(), (repository / RUNNER_RELATIVE_PATH).read_bytes()
    assert result["manifest_binding"]["raw_sha256"] == hashlib.sha256(manifest_raw).hexdigest()
    assert result["source_binding"]["source_sha256"] == {MANIFEST_RELATIVE_PATH: hashlib.sha256(manifest_raw).hexdigest(), RUNNER_RELATIVE_PATH: hashlib.sha256(runner_raw).hexdigest()}
    assert result["content_sha256"] == cloned.result_content_sha256(result)
    assert not {"metric", "metrics", "value", "values", "observations", "results"} & set(result)


def test_unavailable_path_preflights_before_caller_controlled_paths_or_identities(monkeypatch: pytest.MonkeyPatch) -> None:
    class Stop(RuntimeError):
        pass
    def stop() -> dict[str, Any]:
        raise Stop
    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("operation occurred before capacity preflight")
    monkeypatch.setattr(token, "capacity_preflight", stop)
    monkeypatch.setattr(token, "_bound_repository_and_recipe", forbidden)
    monkeypatch.setattr(token, "repository_identity", forbidden)
    monkeypatch.setattr(token.subprocess, "run", forbidden)
    monkeypatch.setattr(token.os, "open", forbidden)
    with pytest.raises(Stop):
        token.run_benchmark(repository=object(), identities=object())


def test_unavailable_artifact_is_deterministic_and_never_reads_identities(prepared_result: PreparedResult) -> None:
    first, cloned, repository, _identity = prepared_result
    class Exploding:
        def __iter__(self) -> Any: raise AssertionError("identities read")
        def __str__(self) -> str: raise AssertionError("identities rendered")
        def __fspath__(self) -> str: raise AssertionError("identities path")
    assert cloned.run_benchmark(repository=repository, identities=Exploding()) == first


@pytest.mark.parametrize("mutation", [
    lambda value: value.update({"metrics": {}}),
    lambda value: value.update({"ran": True}),
    lambda value: value["token_gates"].update({"minimum_complete_board_scan_reduction_percent": 79}),
    lambda value: value["future_required_comparison"]["candidate_arm"].update({"required_supervisor_processes": 11}),
    lambda value: value["repository_binding"].update({"repository_tree": "f" * 40}),
    lambda value: value["source_binding"]["source_sha256"].update({RUNNER_RELATIVE_PATH: "f" * 64}),
    lambda value: value["nonclaims"].reverse(),
])
def test_fake_measurements_stale_bindings_and_weakened_contracts_fail_closed(prepared_result: PreparedResult, mutation: Any) -> None:
    result, cloned, repository, _identity = prepared_result
    candidate = deepcopy(result)
    mutation(candidate)
    _rehash(cloned, candidate)
    with pytest.raises(cloned.TokenBenchmarkError):
        cloned.validate_result(candidate, repository=repository)


@pytest.mark.parametrize("mutation", [
    lambda value: value.update({"baseline_supervisor_processes": 2}),
    lambda value: value.update({"candidate_supervisor_processes": 11}),
    lambda value: value.update({"state_transport": "direct_duckdb"}),
    lambda value: value.update({"direct_database_access_used": True}),
    lambda value: value.update({"ducklake_scheduling_authority_used": True}),
    lambda value: value["zero_tolerance_gate_failures"].update({"tenant_leakage": 1}),
    lambda value: value.update({"repeated_context_token_reduction_percent": 49}),
    lambda value: value.update({"input_tokens_per_accepted_criterion_reduction_percent": 39}),
    lambda value: value.update({"duplicate_model_call_reduction_percent": 59}),
    lambda value: value.update({"eligible_semantic_capsule_reuse_percent": 69}),
    lambda value: value.update({"complete_board_scan_reduction_percent": 79}),
])
def test_dormant_boundary_rejects_noncomparable_unsafe_and_under_threshold_observations(mutation: Any) -> None:
    observation = _valid_observation(token)
    mutation(observation)
    with pytest.raises(token.TokenBenchmarkError):
        token.validate_admitted_token_observation(_valid_admission(token), observation)


@pytest.mark.parametrize("changes", [
    {"baseline_supervisor_processes": 2}, {"candidate_supervisor_processes": 11},
    {"state_authority": "ducklake"}, {"quack_receipt_ref": "token=secret"},
    {"direct_database_access_permitted": True}, {"ducklake_scheduling_authority_permitted": True},
])
def test_dormant_boundary_rejects_forged_or_weakened_admissions(changes: dict[str, Any]) -> None:
    with pytest.raises(token.TokenBenchmarkError):
        token.validate_admitted_token_observation(_valid_admission(token)._replace(**changes), _valid_observation(token))


def test_dormant_execution_boundary_cannot_call_an_executor() -> None:
    class ForbiddenExecutor:
        interface = token.ADMITTED_EXECUTION_INTERFACE
        def execute_token_comparison(self, **_kwargs: Any) -> Any:
            raise AssertionError("executor must be unreachable")
    with pytest.raises(token.TokenCapabilityUnavailable, match="execution is unavailable"):
        token._admitted_execution_boundary(ForbiddenExecutor(), admission=_valid_admission(token))


def test_malformed_and_unsafe_inputs_fail_closed(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema": 1, "schema": 2}', encoding="utf-8")
    with pytest.raises(token.TokenBenchmarkError, match="duplicate JSON key"):
        token._read_object(duplicate)
    linked = tmp_path / "linked.json"
    linked.symlink_to(duplicate)
    with pytest.raises(token.TokenBenchmarkError, match="non-symlink"):
        token._read_object(linked)
    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b"x" * (token._MAX_MANIFEST_BYTES + 1))
    with pytest.raises(token.TokenBenchmarkError, match="byte limit"):
        token._read_object(oversized)


def test_cli_never_reads_identity_file_and_sanitizes_invalid_path(prepared_result: PreparedResult, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    _result, cloned, repository, _identity = prepared_result
    identity_path = tmp_path / "sk-secret-identities.json"
    identity_path.write_text('{"token":"must-not-be-read"}', encoding="utf-8")
    assert cloned.main(["--repository", str(repository), "--identities", str(identity_path)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["execution_status"] == "not_run"
    assert output["metrics_omitted"] is True
    assert cloned.main(["--repository", "/tmp/sk-do-not-echo/repository"]) == 2
    assert "sk-do-not-echo" not in capsys.readouterr().out
