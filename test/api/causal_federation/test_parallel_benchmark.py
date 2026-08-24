"""Closed unavailable-result contracts for the CASF parallel benchmark."""

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
RUNNER_RELATIVE_PATH = "benchmarks/agent_supervisor/causal_event_federation/run_parallel.py"
MANIFEST_RELATIVE_PATH = (
    "benchmarks/agent_supervisor/causal_event_federation/parallel_manifest.json"
)
TEST_RELATIVE_PATH = "test/api/causal_federation/test_parallel_benchmark.py"
RUNNER_PATH = ROOT / RUNNER_RELATIVE_PATH
MANIFEST_PATH = ROOT / MANIFEST_RELATIVE_PATH
MATRIX_PATH = ROOT / "benchmarks/agent_supervisor/causal_event_federation/matrix.yaml"
SPEC = importlib.util.spec_from_file_location("casf_parallel_benchmark", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
parallel = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(parallel)

OWNED_PATHS = (MANIFEST_RELATIVE_PATH, RUNNER_RELATIVE_PATH, TEST_RELATIVE_PATH)
PreparedResult = tuple[dict[str, Any], Any, Path, dict[str, str]]


def _run(*args: str, cwd: Path) -> str:
    return subprocess.run(
        list(args), cwd=cwd, check=True, capture_output=True, text=True
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
        "CASF-039 fixture",
        cwd=repository,
    )
    runner = repository / RUNNER_RELATIVE_PATH
    spec = importlib.util.spec_from_file_location(f"casf_parallel_{path.name}", runner)
    assert spec is not None and spec.loader is not None
    cloned_parallel = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cloned_parallel)
    return repository, cloned_parallel


@pytest.fixture(scope="module")
def prepared_result(tmp_path_factory: pytest.TempPathFactory) -> PreparedResult:
    repository, cloned_parallel = _prepared_clone(tmp_path_factory.mktemp("parallel-unavailable"))
    result = cloned_parallel.run_benchmark(repository=repository, identities=None)
    identity = cloned_parallel.repository_identity(repository)
    return result, cloned_parallel, repository, identity


def _rehash(module: Any, result: dict[str, Any]) -> dict[str, Any]:
    result["content_sha256"] = module.result_content_sha256(result)
    return result


def _all_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value).union(*(_all_keys(child) for child in value.values()), set())
    if isinstance(value, list):
        return set().union(*(_all_keys(child) for child in value), set())
    return set()


def test_manifest_freezes_unavailable_one_and_twelve_process_arms() -> None:
    manifest = parallel.load_manifest(MANIFEST_PATH)

    assert parallel.RESULT_SCHEMA == "casf/parallel-benchmark@1"
    assert parallel.validate_matrix_binding(manifest, MATRIX_PATH) == {
        "relative_path": parallel.MATRIX_RELATIVE_PATH,
        "schema": parallel.MATRIX_SCHEMA,
        "sha256": parallel.MATRIX_SHA256,
    }
    assert manifest["state"] == "capability_unavailable"
    assert manifest["authoritative"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["execution"] == {
        "measurement_scope": "twelve_independent_supervisor_processes_qualified_live_profile",
        "required_supervisor_processes": 12,
        "subprocess_budget": 12,
        "real_independent_processes_required": True,
        "in_process_simulation_qualifies": False,
        "admitted_execution_interface": "CASFParallelAdmittedExecution@1",
        "admitted_execution_available": False,
        "network_permitted": False,
        "direct_database_access_permitted": False,
        "ducklake_scheduling_authority_permitted": False,
        "launch_permitted": False,
    }
    assert tuple(manifest["parallel_comparison"]["future_required_arms"]) == (
        {
            "arm_id": "baseline",
            "required_supervisor_processes": 1,
            "availability": "unavailable",
            "execution_status": "not_run",
            "qualified": False,
        },
        {
            "arm_id": "candidate",
            "required_supervisor_processes": 12,
            "availability": "unavailable",
            "execution_status": "not_run",
            "qualified": False,
        },
    )
    assert manifest["parallel_comparison"]["minimum_accepted_task_throughput_multiplier"] == 3.0
    assert (
        type(manifest["parallel_comparison"]["minimum_accepted_task_throughput_multiplier"])
        is float
    )
    assert tuple(manifest["nonclaims"]) == parallel.NONCLAIMS
    assert tuple(manifest["future_identity_requirements"]) == (
        "repository_commit",
        "repository_tree",
        "control_plane_generation",
        "schema_fingerprint",
        "policy_ref",
        "policy_revision",
        "capability_ref",
        "federation_id",
        "supervisor_id",
        "task_id",
        "attempt_id",
        "worktree_id",
        "assignment_revision",
        "fencing_epoch",
    )


def test_unavailable_result_is_closed_content_addressed_and_tree_bound(
    prepared_result: PreparedResult,
) -> None:
    result, cloned_parallel, repository, identity = prepared_result

    assert (
        cloned_parallel.validate_result(
            result,
            manifest_path=repository / MANIFEST_RELATIVE_PATH,
            matrix_path=repository / cloned_parallel.MATRIX_RELATIVE_PATH,
            current_identity=identity,
        )
        == result
    )
    assert result["schema"] == "casf/parallel-benchmark@1"
    assert result["availability"] == "unavailable"
    assert result["execution_status"] == "not_run"
    assert result["ran"] is False
    assert result["qualified"] is False
    assert result["authoritative"] is False
    assert result["promotion_eligible"] is False
    assert result["metrics_omitted"] is True
    assert result["reason_code"] == cloned_parallel.UNAVAILABLE_REASON_CODE
    assert result["repository_binding"] == {
        **identity,
        "clean": True,
        "observed_before_and_after": True,
    }
    manifest_raw = (repository / MANIFEST_RELATIVE_PATH).read_bytes()
    assert result["manifest_binding"] == {
        "relative_path": MANIFEST_RELATIVE_PATH,
        "schema": cloned_parallel.MANIFEST_SCHEMA,
        "raw_sha256": hashlib.sha256(manifest_raw).hexdigest(),
    }
    assert result["matrix_binding"] == {
        "relative_path": cloned_parallel.MATRIX_RELATIVE_PATH,
        "schema": cloned_parallel.MATRIX_SCHEMA,
        "sha256": cloned_parallel.MATRIX_SHA256,
    }
    assert result["content_sha256"] == cloned_parallel.result_content_sha256(result)
    assert not (
        {"metric", "metrics", "value", "values", "result_ref", "result_refs"} & _all_keys(result)
    )


def test_unavailable_artifact_is_deterministic_and_ignores_caller_identifiers(
    prepared_result: PreparedResult,
) -> None:
    first, cloned_parallel, repository, _identity = prepared_result

    class ExplodingIdentities:
        def __iter__(self) -> Any:
            raise AssertionError("identities must not be iterated")

        def __len__(self) -> int:
            raise AssertionError("identities must not be measured")

        def __getitem__(self, _key: str) -> Any:
            raise AssertionError("identities must not be indexed")

        def __str__(self) -> str:
            raise AssertionError("identities must not be rendered")

    second = cloned_parallel.run_benchmark(
        repository=repository,
        identities=ExplodingIdentities(),  # type: ignore[arg-type]
    )
    assert second == first


def test_capacity_preflight_is_first_before_files_git_or_identity_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class PreflightStop(RuntimeError):
        pass

    def stop() -> dict[str, Any]:
        raise PreflightStop

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("an operation occurred before capacity preflight")

    monkeypatch.setattr(parallel, "_capacity_preflight", stop)
    monkeypatch.setattr(parallel, "_bound_repository_and_recipe", forbidden)
    monkeypatch.setattr(parallel, "repository_identity", forbidden)
    monkeypatch.setattr(parallel, "_read_object", forbidden)
    monkeypatch.setattr(parallel.subprocess, "run", forbidden)
    monkeypatch.setattr(parallel.os, "open", forbidden)
    monkeypatch.setattr(parallel, "_admitted_execution_boundary", forbidden)

    with pytest.raises(PreflightStop):
        parallel.run_benchmark(repository=ROOT, identities=object())  # type: ignore[arg-type]


def test_normal_unavailable_path_only_uses_bounded_git_subprocesses(
    prepared_result: PreparedResult, monkeypatch: pytest.MonkeyPatch
) -> None:
    _result, cloned_parallel, repository, _identity = prepared_result
    actual_run = cloned_parallel.subprocess.run
    commands: list[list[str]] = []

    def checked_run(command: list[str], **kwargs: Any) -> Any:
        assert command[0] == "git"
        assert kwargs["timeout"] == 10
        commands.append(command)
        return actual_run(command, **kwargs)

    def forbidden_boundary(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("dormant execution boundary was reached")

    monkeypatch.setattr(cloned_parallel.subprocess, "run", checked_run)
    monkeypatch.setattr(cloned_parallel, "_admitted_execution_boundary", forbidden_boundary)
    result = cloned_parallel.run_benchmark(repository=repository)

    assert result["execution_status"] == "not_run"
    assert len(commands) == 6
    assert all(command[3] in {"status", "rev-parse"} for command in commands)


def test_dormant_admitted_execution_boundary_cannot_call_an_executor() -> None:
    class ForbiddenExecutor:
        interface = parallel.ADMITTED_EXECUTION_INTERFACE

        def execute(self, **_kwargs: Any) -> Any:
            raise AssertionError("executor must remain unreachable")

    with pytest.raises(
        parallel.ParallelBenchmarkError, match="admitted parallel execution is unavailable"
    ):
        parallel._admitted_execution_boundary(
            ForbiddenExecutor(),
            repository=ROOT,
            manifest=parallel.load_manifest(MANIFEST_PATH),
            matrix_binding=parallel.validate_matrix_binding(
                parallel.load_manifest(MANIFEST_PATH), MATRIX_PATH
            ),
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda value: value["execution"].update({"required_supervisor_processes": 11}),
            "twelve-process bound",
        ),
        (
            lambda value: value["execution"].update({"subprocess_budget": True}),
            "integer",
        ),
        (
            lambda value: value["execution"].update({"launch_permitted": 0}),
            "JSON boolean",
        ),
        (
            lambda value: value["parallel_comparison"]["future_required_arms"][0].update(
                {"required_supervisor_processes": 12}
            ),
            "one baseline and twelve candidates",
        ),
        (
            lambda value: value["parallel_comparison"]["future_required_arms"].reverse(),
            "one baseline and twelve candidates",
        ),
        (
            lambda value: value["parallel_comparison"].update(
                {"minimum_accepted_task_throughput_multiplier": 3}
            ),
            "JSON number 3.0",
        ),
        (
            lambda value: value["parallel_comparison"].update(
                {"minimum_accepted_task_throughput_multiplier": 2.99}
            ),
            "JSON number 3.0",
        ),
        (
            lambda value: value["live_capability"].update({"reason_code": "sk-secret"}),
            "secret-shaped",
        ),
        (
            lambda value: value["live_capability"].update({"reason_code": "x" * 257}),
            "bounded non-empty",
        ),
        (
            lambda value: value["nonclaims"].__setitem__(0, "we measured it"),
            "ordered nonclaims",
        ),
        (
            lambda value: value["nonclaims"].reverse(),
            "ordered nonclaims",
        ),
        (
            lambda value: value["future_identity_requirements"].pop(),
            "future live identity requirements",
        ),
    ],
)
def test_manifest_counts_types_threshold_identifiers_and_prose_are_exact(
    mutation: Any, match: str
) -> None:
    candidate = deepcopy(parallel.load_manifest(MANIFEST_PATH))
    mutation(candidate)

    with pytest.raises(parallel.ParallelBenchmarkError, match=match):
        parallel.validate_manifest(candidate)


def test_duplicate_keys_oversized_documents_and_symlinks_fail_closed(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema": 1, "schema": 2}', encoding="utf-8")
    with pytest.raises(parallel.ParallelBenchmarkError, match="duplicate JSON key"):
        parallel._read_object(duplicate)

    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b"x" * (parallel.MAX_DOCUMENT_BYTES + 1))
    with pytest.raises(parallel.ParallelBenchmarkError, match="byte bound"):
        parallel._read_object(oversized)

    linked = tmp_path / "linked.json"
    linked.symlink_to(duplicate)
    with pytest.raises(parallel.ParallelBenchmarkError, match="cannot open"):
        parallel._read_object(linked)


def test_repository_and_recipe_paths_are_exact_and_repo_contained(
    prepared_result: PreparedResult, tmp_path: Path
) -> None:
    _result, cloned_parallel, repository, _identity = prepared_result
    outside = tmp_path / "parallel_manifest.json"
    outside.write_text("{}", encoding="utf-8")

    with pytest.raises(cloned_parallel.ParallelBenchmarkError, match="exact repository"):
        cloned_parallel.run_benchmark(repository=tmp_path)
    with pytest.raises(cloned_parallel.ParallelBenchmarkError, match="repository-contained"):
        cloned_parallel.run_benchmark(repository=repository, manifest_path=outside)
    with pytest.raises(cloned_parallel.ParallelBenchmarkError, match="repository-contained"):
        cloned_parallel.run_benchmark(repository=repository, matrix_path=outside)
    with pytest.raises(cloned_parallel.ParallelBenchmarkError, match="repository-contained"):
        cloned_parallel.validate_result(
            _result,
            manifest_path=outside,
            matrix_path=repository / cloned_parallel.MATRIX_RELATIVE_PATH,
            current_identity=_identity,
        )


def test_repository_must_remain_stable_across_both_observations(
    prepared_result: PreparedResult, monkeypatch: pytest.MonkeyPatch
) -> None:
    _result, cloned_parallel, repository, identity = prepared_result
    observations = [identity, {**identity, "repository_tree": "f" * 40}]

    monkeypatch.setattr(
        cloned_parallel, "repository_identity", lambda _repository: observations.pop(0)
    )
    with pytest.raises(cloned_parallel.ParallelBenchmarkError, match="changed during"):
        cloned_parallel.run_benchmark(repository=repository)


@pytest.mark.parametrize(
    ("mutation", "rehash", "match"),
    [
        (lambda value: value.update({"metrics": {"throughput": 0}}), True, "forbidden"),
        (
            lambda value: value["repository_binding"].update({"values": {"throughput": 0}}),
            True,
            "forbidden",
        ),
        (lambda value: value.update({"ran": True}), True, "result.ran has changed"),
        (
            lambda value: value.update({"metrics_omitted": False}),
            True,
            "metrics_omitted has changed",
        ),
        (
            lambda value: value["future_required_arms"][1].update(
                {"required_supervisor_processes": 1}
            ),
            True,
            "one baseline and twelve candidates",
        ),
        (
            lambda value: value["comparison_requirement"].update(
                {"minimum_accepted_task_throughput_multiplier": 3}
            ),
            True,
            "JSON number 3.0",
        ),
        (
            lambda value: value["manifest_binding"].update({"raw_sha256": "f" * 64}),
            True,
            "raw manifest binding",
        ),
        (
            lambda value: value["matrix_binding"].update({"sha256": "f" * 64}),
            True,
            "matrix binding",
        ),
        (lambda value: value.update({"content_sha256": "f" * 64}), False, "content address"),
    ],
)
def test_fake_measurements_wrong_types_and_stale_content_fail_closed(
    prepared_result: PreparedResult, mutation: Any, rehash: bool, match: str
) -> None:
    result, cloned_parallel, repository, identity = prepared_result
    candidate = deepcopy(result)
    mutation(candidate)
    if rehash:
        _rehash(cloned_parallel, candidate)

    with pytest.raises(cloned_parallel.ParallelBenchmarkError, match=match):
        cloned_parallel.validate_result(
            candidate,
            manifest_path=repository / MANIFEST_RELATIVE_PATH,
            matrix_path=repository / cloned_parallel.MATRIX_RELATIVE_PATH,
            current_identity=identity,
        )


def test_replay_observes_current_repository_and_rejects_forged_identity_assertions(
    prepared_result: PreparedResult,
) -> None:
    result, cloned_parallel, repository, identity = prepared_result
    arguments = {
        "manifest_path": repository / MANIFEST_RELATIVE_PATH,
        "matrix_path": repository / cloned_parallel.MATRIX_RELATIVE_PATH,
    }
    assert cloned_parallel.validate_result(result, **arguments) == result
    with pytest.raises(cloned_parallel.ParallelBenchmarkError, match="changed during"):
        cloned_parallel.validate_result(
            result,
            current_identity={**identity, "repository_tree": "f" * 40},
            **arguments,
        )

    forged = deepcopy(result)
    forged_identity = {
        "repository_commit": "e" * 40,
        "repository_tree": "f" * 40,
    }
    forged["repository_binding"].update(forged_identity)
    _rehash(cloned_parallel, forged)
    with pytest.raises(cloned_parallel.ParallelBenchmarkError, match="stale for the current tree"):
        cloned_parallel.validate_result(forged, **arguments)
    with pytest.raises(cloned_parallel.ParallelBenchmarkError, match="changed during"):
        cloned_parallel.validate_result(
            forged,
            current_identity=forged_identity,
            **arguments,
        )

    class TextSubclass(str):
        pass

    wrong_type = deepcopy(result)
    wrong_type["schema"] = TextSubclass(cloned_parallel.RESULT_SCHEMA)
    _rehash(cloned_parallel, wrong_type)
    with pytest.raises(cloned_parallel.ParallelBenchmarkError, match="bounded non-empty string"):
        cloned_parallel.validate_result(wrong_type, **arguments)


def test_cli_distinguishes_valid_unavailable_from_invalid_and_never_reads_identity_file(
    prepared_result: PreparedResult,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _result, cloned_parallel, repository, _identity = prepared_result
    identity_path = tmp_path / "sk-secret-identities.json"
    identity_path.write_text('{"api_key":"must-not-be-read"}', encoding="utf-8")
    actual_read = cloned_parallel._read_object

    def checked_read(path: Path | str) -> dict[str, Any]:
        assert Path(path) != identity_path
        return actual_read(path)

    monkeypatch.setattr(cloned_parallel, "_read_object", checked_read)
    assert (
        cloned_parallel.main(["--repository", str(repository), "--identities", str(identity_path)])
        == 0
    )
    available_output = capsys.readouterr().out
    unavailable = json.loads(available_output)
    assert unavailable["schema"] == cloned_parallel.RESULT_SCHEMA
    assert unavailable["availability"] == "unavailable"
    assert unavailable["execution_status"] == "not_run"
    assert "must-not-be-read" not in available_output
    assert "sk-secret" not in available_output

    assert cloned_parallel.main([]) == 2
    invalid = json.loads(capsys.readouterr().out)
    assert invalid == {
        "schema": cloned_parallel.ERROR_SCHEMA,
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
    assert parallel.main(["--repository", secret_path]) == 2
    raw = capsys.readouterr().out
    invalid = json.loads(raw)
    assert invalid["schema"] == parallel.ERROR_SCHEMA
    assert invalid["execution_status"] == "invalid"
    assert "sk-do-not-echo" not in raw


def test_cli_symlink_loops_have_stable_sanitized_repository_and_recipe_errors(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    first = tmp_path / "sk-secret-loop-a"
    second = tmp_path / "sk-secret-loop-b"
    first.symlink_to(second.name)
    second.symlink_to(first.name)
    cases = (
        (
            ["--repository", str(first)],
            {
                "schema": parallel.ERROR_SCHEMA,
                "execution_status": "invalid",
                "error_code": "invalid_repository",
                "message": "repository does not contain this benchmark runner",
            },
        ),
        (
            ["--repository", str(ROOT), "--manifest", str(first)],
            {
                "schema": parallel.ERROR_SCHEMA,
                "execution_status": "invalid",
                "error_code": "invalid_recipe_path",
                "message": "manifest or matrix path is outside the canonical recipe",
            },
        ),
        (
            ["--repository", str(ROOT), "--matrix", str(first)],
            {
                "schema": parallel.ERROR_SCHEMA,
                "execution_status": "invalid",
                "error_code": "invalid_recipe_path",
                "message": "manifest or matrix path is outside the canonical recipe",
            },
        ),
    )

    for arguments, expected in cases:
        assert parallel.main(arguments) == 2
        raw = capsys.readouterr().out
        assert json.loads(raw) == expected
        assert str(first) not in raw
        assert "sk-secret-loop" not in raw
