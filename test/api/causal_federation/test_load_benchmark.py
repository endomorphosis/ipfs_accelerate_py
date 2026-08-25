"""Closed unavailable and dormant-boundary contracts for CASF-040."""

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
RUNNER_RELATIVE_PATH = "benchmarks/agent_supervisor/causal_event_federation/run_load.py"
MANIFEST_RELATIVE_PATH = "benchmarks/agent_supervisor/causal_event_federation/load_manifest.json"
TEST_RELATIVE_PATH = "test/api/causal_federation/test_load_benchmark.py"
RUNNER_PATH = ROOT / RUNNER_RELATIVE_PATH
MANIFEST_PATH = ROOT / MANIFEST_RELATIVE_PATH
MATRIX_PATH = ROOT / "benchmarks/agent_supervisor/causal_event_federation/matrix.yaml"
SPEC = importlib.util.spec_from_file_location("casf_load_benchmark", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
load = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(load)

OWNED_PATHS = (MANIFEST_RELATIVE_PATH, RUNNER_RELATIVE_PATH, TEST_RELATIVE_PATH)
PreparedResult = tuple[dict[str, Any], Any, Path, dict[str, str]]


def _run(*args: str, cwd: Path) -> str:
    return subprocess.run(
        list(args),
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _prepared_clone(path: Path) -> tuple[Path, Any]:
    repository = path / "repository"
    _run("git", "clone", "-q", "--shared", str(ROOT), str(repository), cwd=path)
    for relative_path in OWNED_PATHS:
        destination = repository / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative_path, destination)
    _run("git", "add", *OWNED_PATHS, cwd=repository)
    _run(
        "git",
        "-c",
        "user.name=CASF Test",
        "-c",
        "user.email=casf-test@example.invalid",
        "commit",
        "-q",
        "--allow-empty",
        "-m",
        "CASF-040 fixture",
        cwd=repository,
    )
    runner = repository / RUNNER_RELATIVE_PATH
    spec = importlib.util.spec_from_file_location(f"casf_load_{path.name}", runner)
    assert spec is not None and spec.loader is not None
    cloned_load = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cloned_load)
    return repository, cloned_load


@pytest.fixture(scope="module")
def prepared_result(tmp_path_factory: pytest.TempPathFactory) -> PreparedResult:
    repository, cloned_load = _prepared_clone(tmp_path_factory.mktemp("load-unavailable"))
    result = cloned_load.run_benchmark(repository=repository, identities=None)
    identity = cloned_load.repository_identity(repository)
    return result, cloned_load, repository, identity


def _rehash(module: Any, result: dict[str, Any]) -> dict[str, Any]:
    result["content_sha256"] = module.result_content_sha256(result)
    return result


def _all_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value).union(*(_all_keys(child) for child in value.values()), set())
    if isinstance(value, list):
        return set().union(*(_all_keys(child) for child in value), set())
    return set()


def _valid_admission(module: Any) -> Any:
    return module.LoadAdmission(
        schema=module.LIVE_ATTESTATION_SCHEMA,
        supervisor_processes=12,
        registered_logical_agents=256,
        maximum_concurrent_subagents=64,
        minimum_bounded_tasks=1000,
        minimum_event_deliveries_with_replay=100000,
        replay_required=True,
        state_authority="authenticated_typed_quack",
        quack_receipt_ref=f"sha256:{'a' * 64}",
        direct_database_access_permitted=False,
        ducklake_scheduling_authority_permitted=False,
    )


def _valid_observation(module: Any) -> dict[str, Any]:
    return {
        "schema": module.LIVE_OBSERVATION_SCHEMA,
        "supervisor_process_ids": list(range(1001, 1013)),
        "logical_agent_ids": [f"agent:{index:03d}" for index in range(256)],
        "maximum_active_subagents": 64,
        "bounded_tasks_completed": 1000,
        "event_deliveries_observed": 100000,
        "replay_deliveries_observed": 1,
        "event_delivery_count_includes_replay": True,
        "lost_deliveries": 0,
        "duplicate_committed_effects": 0,
        "zero_tolerance_gate_failures": {gate: 0 for gate in module.ZERO_TOLERANCE_GATES},
        "state_transport": "authenticated_typed_quack",
        "quack_receipt_ref": f"sha256:{'a' * 64}",
        "direct_database_access_used": False,
        "ducklake_scheduling_authority_used": False,
    }


def test_manifest_freezes_exact_load_profile_authority_and_unavailability() -> None:
    manifest = load.load_manifest(MANIFEST_PATH)

    assert load.RESULT_SCHEMA == "casf/load-benchmark@1"
    assert load.validate_matrix_binding(manifest, MATRIX_PATH) == {
        "relative_path": load.MATRIX_RELATIVE_PATH,
        "schema": load.MATRIX_SCHEMA,
        "sha256": load.MATRIX_SHA256,
    }
    assert manifest["state"] == "capability_unavailable"
    assert manifest["authoritative"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["execution"] == {
        "measurement_scope": (
            "twelve_independent_supervisor_processes_256_registered_agents_required_live_profile"
        ),
        "required_supervisor_processes": 12,
        "registered_logical_agents": 256,
        "maximum_concurrent_subagents": 64,
        "minimum_bounded_tasks": 1000,
        "minimum_event_deliveries_with_replay": 100000,
        "subprocess_budget": 12,
        "real_independent_processes_required": True,
        "in_process_simulation_qualifies": False,
        "admitted_execution_interface": "CASFLoadAdmittedExecution@1",
        "admitted_execution_available": False,
        "network_permitted": False,
        "authenticated_typed_quack_required": True,
        "direct_database_access_permitted": False,
        "ducklake_scheduling_authority_permitted": False,
        "ducklake_projection_authoritative": False,
        "launch_permitted": False,
    }
    assert manifest["capacity_preflight"] == load.capacity_preflight()
    assert manifest["live_capability"] == {
        "availability": "unavailable",
        "execution_status": "not_run",
        "ran": False,
        "qualified": False,
        "reason_code": load.REASON_CODE,
        "required_attestation": load.LIVE_ATTESTATION_SCHEMA,
        "required_evidence": (
            "current_generation_accepted_gate_current_fences_and_live_"
            "host_provider_proof_merge_storage_telemetry"
        ),
        "metrics_omitted": True,
    }
    assert tuple(manifest["zero_tolerance_gates"]) == load.ZERO_TOLERANCE_GATES
    assert tuple(manifest["future_identity_requirements"]) == load.REQUIRED_IDENTITIES
    assert tuple(manifest["nonclaims"]) == load.NONCLAIMS


def test_unavailable_result_is_closed_content_addressed_and_source_bound(
    prepared_result: PreparedResult,
) -> None:
    result, cloned_load, repository, identity = prepared_result

    assert (
        cloned_load.validate_result(
            result,
            repository=repository,
            manifest_path=repository / MANIFEST_RELATIVE_PATH,
            matrix_path=repository / cloned_load.MATRIX_RELATIVE_PATH,
        )
        == result
    )
    assert result["schema"] == "casf/load-benchmark@1"
    assert result["availability"] == "unavailable"
    assert result["execution_status"] == "not_run"
    assert result["ran"] is False
    assert result["qualified"] is False
    assert result["metrics_omitted"] is True
    assert result["authoritative"] is False
    assert result["promotion_eligible"] is False
    assert result["repository_binding"] == {
        **identity,
        "clean": True,
        "observed_before_and_after": True,
    }
    manifest_raw = (repository / MANIFEST_RELATIVE_PATH).read_bytes()
    runner_raw = (repository / RUNNER_RELATIVE_PATH).read_bytes()
    assert result["manifest_binding"] == {
        "relative_path": MANIFEST_RELATIVE_PATH,
        "schema": cloned_load.MANIFEST_SCHEMA,
        "raw_sha256": hashlib.sha256(manifest_raw).hexdigest(),
    }
    assert result["matrix_binding"] == {
        "relative_path": cloned_load.MATRIX_RELATIVE_PATH,
        "schema": cloned_load.MATRIX_SCHEMA,
        "sha256": cloned_load.MATRIX_SHA256,
    }
    assert result["source_binding"] == {
        "source_sha256": {
            MANIFEST_RELATIVE_PATH: hashlib.sha256(manifest_raw).hexdigest(),
            RUNNER_RELATIVE_PATH: hashlib.sha256(runner_raw).hexdigest(),
        },
        "observed_before_and_after": True,
    }
    assert result["content_sha256"] == cloned_load.result_content_sha256(result)
    assert not (
        {"metric", "metrics", "value", "values", "observations", "result_values", "error"}
        & _all_keys(result)
    )


def test_unavailable_artifact_is_deterministic_and_never_reads_caller_identities(
    prepared_result: PreparedResult,
) -> None:
    first, cloned_load, repository, _identity = prepared_result

    class ExplodingIdentities:
        def __iter__(self) -> Any:
            raise AssertionError("identities must not be iterated")

        def __len__(self) -> int:
            raise AssertionError("identities must not be measured")

        def __getitem__(self, _key: str) -> Any:
            raise AssertionError("identities must not be indexed")

        def __str__(self) -> str:
            raise AssertionError("identities must not be rendered")

        def __fspath__(self) -> str:
            raise AssertionError("identities must not be treated as a path")

    second = cloned_load.run_benchmark(
        repository=repository,
        identities=ExplodingIdentities(),
    )
    assert second == first


def test_capacity_preflight_is_first_before_paths_git_reads_or_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class PreflightStop(RuntimeError):
        pass

    def stop() -> dict[str, Any]:
        raise PreflightStop

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("an operation occurred before capacity preflight")

    monkeypatch.setattr(load, "capacity_preflight", stop)
    monkeypatch.setattr(load, "_bound_repository_and_recipe", forbidden)
    monkeypatch.setattr(load, "repository_identity", forbidden)
    monkeypatch.setattr(load, "_read_bounded_regular_bytes", forbidden)
    monkeypatch.setattr(load.subprocess, "run", forbidden)
    monkeypatch.setattr(load.os, "open", forbidden)
    monkeypatch.setattr(load, "_admitted_execution_boundary", forbidden)

    with pytest.raises(PreflightStop):
        load.run_benchmark(repository=object(), identities=object())


def test_normal_unavailable_path_only_spawns_bounded_git_identity_commands(
    prepared_result: PreparedResult,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _result, cloned_load, repository, _identity = prepared_result
    actual_run = cloned_load.subprocess.run
    commands: list[list[str]] = []

    def checked_run(command: list[str], **kwargs: Any) -> Any:
        assert command[0] == "git"
        assert kwargs["timeout"] == 15
        commands.append(command)
        return actual_run(command, **kwargs)

    def forbidden_boundary(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("dormant execution boundary was reached")

    monkeypatch.setattr(cloned_load.subprocess, "run", checked_run)
    monkeypatch.setattr(cloned_load, "_admitted_execution_boundary", forbidden_boundary)
    result = cloned_load.run_benchmark(repository=repository)

    assert result["execution_status"] == "not_run"
    assert len(commands) == 6
    assert all(command[3] in {"status", "rev-parse"} for command in commands)


def test_dormant_contract_enforces_exact_profile_without_self_qualifying() -> None:
    admission = _valid_admission(load)
    observation = _valid_observation(load)
    checked = load.validate_admitted_load_observation(admission, observation)
    assert checked == observation
    assert "qualified" not in checked
    assert "authoritative" not in checked
    assert "promotion_eligible" not in checked


def test_dormant_admitted_execution_boundary_never_invokes_executor() -> None:
    class ForbiddenExecutor:
        interface = load.ADMITTED_EXECUTION_INTERFACE

        def execute_load(self, **_kwargs: Any) -> Any:
            raise AssertionError("executor must remain unreachable at current capacity")

    with pytest.raises(load.LoadCapabilityUnavailable, match="execution is unavailable"):
        load._admitted_execution_boundary(
            ForbiddenExecutor(),
            admission=_valid_admission(load),
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["supervisor_process_ids"].pop(),
        lambda value: value["supervisor_process_ids"].__setitem__(11, 1001),
        lambda value: value["logical_agent_ids"].pop(),
        lambda value: value["logical_agent_ids"].__setitem__(255, "agent:000"),
        lambda value: value.update({"maximum_active_subagents": 65}),
        lambda value: value.update({"bounded_tasks_completed": 999}),
        lambda value: value.update({"event_deliveries_observed": 99999}),
        lambda value: value.update({"replay_deliveries_observed": 0}),
        lambda value: value.update({"event_delivery_count_includes_replay": False}),
        lambda value: value.update({"lost_deliveries": 1}),
        lambda value: value.update({"duplicate_committed_effects": 1}),
        lambda value: value["zero_tolerance_gate_failures"].update({"tenant_leakage": 1}),
        lambda value: value.update({"state_transport": "direct_duckdb"}),
        lambda value: value.update({"quack_receipt_ref": f"sha256:{'b' * 64}"}),
        lambda value: value.update({"direct_database_access_used": True}),
        lambda value: value.update({"ducklake_scheduling_authority_used": True}),
    ],
)
def test_dormant_boundary_rejects_under_capacity_loss_replay_and_authority_violations(
    mutation: Any,
) -> None:
    admission = _valid_admission(load)
    observation = _valid_observation(load)
    mutation(observation)

    with pytest.raises(load.LoadBenchmarkError):
        load.validate_admitted_load_observation(admission, observation)


@pytest.mark.parametrize(
    "changes",
    [
        {"supervisor_processes": 1},
        {"registered_logical_agents": 255},
        {"maximum_concurrent_subagents": 65},
        {"minimum_bounded_tasks": 999},
        {"minimum_event_deliveries_with_replay": 99999},
        {"replay_required": 1},
        {"state_authority": "ducklake"},
        {"direct_database_access_permitted": True},
        {"ducklake_scheduling_authority_permitted": True},
        {"quack_receipt_ref": "token=secret"},
    ],
)
def test_dormant_boundary_rejects_forged_or_weakened_admissions(changes: dict[str, Any]) -> None:
    admission = _valid_admission(load)._replace(**changes)

    with pytest.raises(load.LoadBenchmarkError):
        load.validate_admitted_load_observation(admission, _valid_observation(load))


def test_dormant_contract_rejects_equality_objects_string_subclasses_and_github_tokens() -> None:
    class EqualQuack:
        def __eq__(self, other: Any) -> bool:
            return other == "authenticated_typed_quack"

    class TextSubclass(str):
        pass

    observation = _valid_observation(load)
    observation["state_transport"] = EqualQuack()
    with pytest.raises(load.LoadBenchmarkError, match="exact text"):
        load.validate_admitted_load_observation(_valid_admission(load), observation)

    observation = _valid_observation(load)
    observation["logical_agent_ids"][0] = TextSubclass("agent:000")
    with pytest.raises(load.LoadBenchmarkError, match="exact text"):
        load.validate_admitted_load_observation(_valid_admission(load), observation)

    observation = _valid_observation(load)
    observation["quack_receipt_ref"] = TextSubclass(f"sha256:{'a' * 64}")
    with pytest.raises(load.LoadBenchmarkError, match="exact text"):
        load.validate_admitted_load_observation(_valid_admission(load), observation)

    admission = _valid_admission(load)._replace(
        quack_receipt_ref=TextSubclass(f"sha256:{'a' * 64}")
    )
    with pytest.raises(load.LoadBenchmarkError, match="exact text"):
        load.validate_admitted_load_observation(admission, _valid_observation(load))

    observation = _valid_observation(load)
    observation["logical_agent_ids"][0] = "github_pat_0123456789abcdef"
    with pytest.raises(load.LoadBenchmarkError, match="credential-shaped"):
        load.validate_admitted_load_observation(_valid_admission(load), observation)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["execution"].update({"required_supervisor_processes": 12.0}),
        lambda value: value["execution"].update({"registered_logical_agents": 256.0}),
        lambda value: value["execution"].update({"maximum_concurrent_subagents": True}),
        lambda value: value["execution"].update({"minimum_bounded_tasks": 999}),
        lambda value: value["execution"].update({"minimum_event_deliveries_with_replay": 99999}),
        lambda value: value["execution"].update({"launch_permitted": 0}),
        lambda value: value["execution"].update({"authenticated_typed_quack_required": 1}),
        lambda value: value["capacity_preflight"].update({"current_supervisor_processes": 2}),
        lambda value: value["capacity_preflight"].update(
            {"authenticated_typed_quack_live_capacity": True}
        ),
        lambda value: value["live_capability"].update({"ran": 0}),
        lambda value: value["live_capability"].update({"qualified": 0}),
        lambda value: value["live_capability"].update({"metrics_omitted": 1}),
        lambda value: value["live_capability"].update({"reason_code": "token=secret"}),
        lambda value: value["nonclaims"].__setitem__(0, "we measured it"),
        lambda value: value["nonclaims"].reverse(),
        lambda value: value.update({"result_storage": "write it anywhere"}),
        lambda value: value["future_identity_requirements"].pop(),
    ],
)
def test_manifest_counts_types_authority_and_pinned_prose_are_exact(mutation: Any) -> None:
    candidate = deepcopy(load.load_manifest(MANIFEST_PATH))
    mutation(candidate)

    with pytest.raises(load.LoadBenchmarkError):
        load.validate_manifest(candidate)


def test_duplicate_nonfinite_oversized_invalid_utf8_and_symlink_inputs_fail_closed(
    tmp_path: Path,
) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema": 1, "schema": 2}', encoding="utf-8")
    with pytest.raises(load.LoadBenchmarkError, match="duplicate JSON key"):
        load._read_object(duplicate)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"value": NaN}', encoding="utf-8")
    with pytest.raises(load.LoadBenchmarkError, match="non-finite"):
        load._read_object(nonfinite)

    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b"x" * (load._MAX_MANIFEST_BYTES + 1))
    with pytest.raises(load.LoadBenchmarkError, match="byte limit"):
        load._read_object(oversized)

    invalid_utf8 = tmp_path / "invalid.json"
    invalid_utf8.write_bytes(b"\xff")
    with pytest.raises(load.LoadBenchmarkError, match="UTF-8"):
        load._read_object(invalid_utf8)

    linked = tmp_path / "linked.json"
    linked.symlink_to(duplicate)
    with pytest.raises(load.LoadBenchmarkError, match="non-symlink"):
        load._read_object(linked)


def test_future_identity_contract_rejects_wrong_types_task_and_secrets() -> None:
    identities = {
        "repository_commit": "a" * 40,
        "repository_tree": "b" * 40,
        "control_plane_generation": 1,
        "schema_fingerprint": "schema:fixture",
        "policy_ref": "policy:fixture",
        "policy_revision": "policy-revision:fixture",
        "capability_ref": "capability:fixture",
        "federation_id": "federation:fixture",
        "supervisor_id": "supervisor:fixture",
        "task_id": "CASF-040",
        "attempt_id": "attempt:fixture",
        "worktree_id": "worktree:fixture",
        "assignment_revision": 1,
        "fencing_epoch": 1,
    }
    assert load._validate_identities(identities) == identities

    wrong_type = {**identities, "fencing_epoch": True}
    with pytest.raises(load.LoadBenchmarkError, match="integer"):
        load._validate_identities(wrong_type)
    wrong_task = {**identities, "task_id": "CASF-039"}
    with pytest.raises(load.LoadBenchmarkError, match="CASF-040"):
        load._validate_identities(wrong_task)
    secret = {**identities, "policy_ref": "token=must-not-land"}
    with pytest.raises(load.LoadBenchmarkError, match="credential-shaped"):
        load._validate_identities(secret)
    github_secret = {**identities, "policy_ref": "github_pat_0123456789abcdef"}
    with pytest.raises(load.LoadBenchmarkError, match="credential-shaped"):
        load._validate_identities(github_secret)
    padded = {**identities, "capability_ref": " capability:fixture"}
    with pytest.raises(load.LoadBenchmarkError, match="exact text"):
        load._validate_identities(padded)


def test_repository_and_recipe_paths_are_exact_regular_and_repo_contained(
    prepared_result: PreparedResult,
    tmp_path: Path,
) -> None:
    result, cloned_load, repository, _identity = prepared_result
    outside = tmp_path / "load_manifest.json"
    outside.write_text("{}", encoding="utf-8")
    linked = tmp_path / "linked-manifest.json"
    linked.symlink_to(repository / MANIFEST_RELATIVE_PATH)

    with pytest.raises(cloned_load.LoadBenchmarkError, match="exact benchmark runner"):
        cloned_load.run_benchmark(repository=tmp_path)
    with pytest.raises(cloned_load.LoadBenchmarkError, match="exact measured-tree"):
        cloned_load.run_benchmark(repository=repository, manifest_path=outside)
    with pytest.raises(cloned_load.LoadBenchmarkError, match="exact measured-tree"):
        cloned_load.run_benchmark(repository=repository, matrix_path=outside)
    with pytest.raises(cloned_load.LoadBenchmarkError, match="symlink"):
        cloned_load.run_benchmark(repository=repository, manifest_path=linked)
    with pytest.raises(cloned_load.LoadBenchmarkError, match="exact measured-tree"):
        cloned_load.validate_result(
            result,
            repository=repository,
            manifest_path=outside,
            matrix_path=repository / cloned_load.MATRIX_RELATIVE_PATH,
        )


def test_repository_and_sources_must_remain_stable_across_observations(
    prepared_result: PreparedResult,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _result, cloned_load, repository, identity = prepared_result
    observations = [identity, {**identity, "repository_tree": "f" * 40}]
    monkeypatch.setattr(
        cloned_load,
        "repository_identity",
        lambda _repository: observations.pop(0),
    )
    with pytest.raises(cloned_load.LoadBenchmarkError, match="changed during"):
        cloned_load.run_benchmark(repository=repository)


def test_source_byte_change_during_snapshot_fails_closed(
    prepared_result: PreparedResult,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _result, cloned_load, repository, _identity = prepared_result
    actual_read = cloned_load._read_bounded_regular_bytes
    calls = 0

    def changing_read(*args: Any, **kwargs: Any) -> bytes:
        nonlocal calls
        calls += 1
        raw = actual_read(*args, **kwargs)
        return raw + b"# changed" if calls == 5 else raw

    monkeypatch.setattr(cloned_load, "_read_bounded_regular_bytes", changing_read)
    with pytest.raises(cloned_load.LoadBenchmarkError, match="changed during"):
        cloned_load.run_benchmark(repository=repository)


@pytest.mark.parametrize(
    ("mutation", "rehash"),
    [
        (lambda value: value.update({"metrics": {"deliveries": 0}}), True),
        (lambda value: value.update({"ran": True}), True),
        (lambda value: value.update({"ran": 0}), True),
        (
            lambda value: value["future_required_profile"].update(
                {"registered_logical_agents": 255}
            ),
            True,
        ),
        (
            lambda value: value["capability_preflight"].update(
                {"current_registered_logical_agents": 256}
            ),
            True,
        ),
        (
            lambda value: value["manifest_binding"].update({"raw_sha256": "f" * 64}),
            True,
        ),
        (lambda value: value["matrix_binding"].update({"sha256": "f" * 64}), True),
        (
            lambda value: value["source_binding"]["source_sha256"].update(
                {RUNNER_RELATIVE_PATH: "f" * 64}
            ),
            True,
        ),
        (
            lambda value: value["repository_binding"].update({"repository_tree": "f" * 40}),
            True,
        ),
        (lambda value: value["nonclaims"].reverse(), True),
        (lambda value: value.update({"content_sha256": "f" * 64}), False),
    ],
)
def test_fake_measurements_wrong_types_and_stale_bindings_fail_closed(
    prepared_result: PreparedResult,
    mutation: Any,
    rehash: bool,
) -> None:
    result, cloned_load, repository, _identity = prepared_result
    candidate = deepcopy(result)
    mutation(candidate)
    if rehash:
        _rehash(cloned_load, candidate)

    with pytest.raises(cloned_load.LoadBenchmarkError):
        cloned_load.validate_result(
            candidate,
            repository=repository,
            manifest_path=repository / MANIFEST_RELATIVE_PATH,
            matrix_path=repository / cloned_load.MATRIX_RELATIVE_PATH,
        )


def test_dirty_repository_invalidates_replay(
    tmp_path: Path,
) -> None:
    repository, cloned_load = _prepared_clone(tmp_path)
    result = cloned_load.run_benchmark(repository=repository)
    (repository / "uncommitted.txt").write_text("dirty\n", encoding="utf-8")

    with pytest.raises(cloned_load.LoadBenchmarkError, match="dirty"):
        cloned_load.validate_result(result, repository=repository)


def test_cli_distinguishes_valid_unavailable_from_invalid_and_never_reads_identity_file(
    prepared_result: PreparedResult,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _result, cloned_load, repository, _identity = prepared_result
    identity_path = tmp_path / "sk-secret-identities.json"
    identity_path.write_text('{"api_key":"must-not-be-read"}', encoding="utf-8")
    actual_read = cloned_load._read_bounded_regular_bytes

    def checked_read(path: Path | str, **kwargs: Any) -> bytes:
        assert Path(path) != identity_path
        return actual_read(path, **kwargs)

    monkeypatch.setattr(cloned_load, "_read_bounded_regular_bytes", checked_read)
    assert (
        cloned_load.main(["--repository", str(repository), "--identities", str(identity_path)]) == 0
    )
    raw = capsys.readouterr().out
    unavailable = json.loads(raw)
    assert unavailable["schema"] == cloned_load.RESULT_SCHEMA
    assert unavailable["availability"] == "unavailable"
    assert unavailable["execution_status"] == "not_run"
    assert unavailable["ran"] is False
    assert unavailable["qualified"] is False
    assert unavailable["metrics_omitted"] is True
    assert "must-not-be-read" not in raw
    assert "sk-secret" not in raw

    assert cloned_load.main([]) == 2
    invalid = json.loads(capsys.readouterr().out)
    assert invalid == {
        "schema": cloned_load.ERROR_SCHEMA,
        "execution_status": "invalid",
        "error_code": "missing_required_argument",
        "message": "repository is required",
    }
    assert "availability" not in invalid
    assert "qualified" not in invalid


def test_cli_invalid_path_does_not_echo_secret_shaped_input(
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret_path = "/tmp/sk-do-not-echo/repository"
    assert load.main(["--repository", secret_path]) == 2
    raw = capsys.readouterr().out
    invalid = json.loads(raw)
    assert invalid["schema"] == load.ERROR_SCHEMA
    assert invalid["execution_status"] == "invalid"
    assert "sk-do-not-echo" not in raw


def test_cli_symlink_loops_have_stable_sanitized_path_errors(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    first = tmp_path / "github_pat_secret-loop-a"
    second = tmp_path / "github_pat_secret-loop-b"
    first.symlink_to(second.name)
    second.symlink_to(first.name)
    expected = {
        "schema": load.ERROR_SCHEMA,
        "execution_status": "invalid",
        "error_code": "unsafe_source_path",
        "message": "a required source path is unavailable or unsafe",
    }
    cases = (
        ["--repository", str(first)],
        ["--repository", str(ROOT), "--manifest", str(first)],
        ["--repository", str(ROOT), "--matrix", str(first)],
    )

    for arguments in cases:
        assert load.main(arguments) == 2
        raw = capsys.readouterr().out
        assert json.loads(raw) == expected
        assert str(first) not in raw
        assert "github_pat_secret-loop" not in raw
