"""Closed contracts and real-process evidence for the CASF idle baseline."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import threading
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
RUNNER_PATH = ROOT / "benchmarks/agent_supervisor/causal_event_federation/run_idle.py"
MANIFEST_PATH = ROOT / "benchmarks/agent_supervisor/causal_event_federation/idle_manifest.json"
MATRIX_PATH = ROOT / "benchmarks/agent_supervisor/causal_event_federation/matrix.yaml"
SPEC = importlib.util.spec_from_file_location("casf_idle_benchmark", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
idle = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(idle)

OWNED_PATHS = (
    "benchmarks/agent_supervisor/causal_event_federation/idle_manifest.json",
    "benchmarks/agent_supervisor/causal_event_federation/run_idle.py",
    "test/api/causal_federation/test_idle_benchmark.py",
)
MeasuredResult = tuple[dict[str, Any], dict[str, str], Path]


def _run(*args: str, cwd: Path) -> str:
    return subprocess.run(
        list(args), cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def _clean_repository(path: Path) -> dict[str, str]:
    path.mkdir()
    _run("git", "init", "-q", cwd=path)
    (path / "measured.txt").write_text("CASF-038 real-process fixture\n", encoding="utf-8")
    _run("git", "add", "measured.txt", cwd=path)
    _run(
        "git",
        "-c",
        "user.name=CASF Test",
        "-c",
        "user.email=casf-test@example.invalid",
        "commit",
        "-q",
        "-m",
        "fixture",
        cwd=path,
    )
    return idle.repository_identity(path)


def _prepared_measured_clone(path: Path) -> tuple[Path, Any]:
    repository_path = path / "repository"
    _run("git", "clone", "-q", "--shared", str(ROOT), str(repository_path), cwd=path)
    for relative_path in OWNED_PATHS:
        target = repository_path / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative_path, target)
    _run("git", "add", *OWNED_PATHS, cwd=repository_path)
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
        "CASF-038 fixture",
        cwd=repository_path,
    )
    runner_path = repository_path / OWNED_PATHS[1]
    spec = importlib.util.spec_from_file_location(
        f"casf_idle_benchmark_clone_{path.name}", runner_path
    )
    assert spec is not None and spec.loader is not None
    cloned_idle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cloned_idle)
    return repository_path, cloned_idle


def _identities(repository: dict[str, str]) -> dict[str, Any]:
    return {
        **repository,
        "control_plane_generation": 7,
        "schema_fingerprint": "schema:fixture",
        "policy_ref": "policy:fixture",
        "policy_revision": "policy-revision:fixture",
        "capability_ref": "capability:fixture",
        "federation_id": "federation:fixture",
        "supervisor_id": "supervisor:fixture",
        "task_id": "CASF-038",
        "attempt_id": "attempt:fixture",
        "worktree_id": "worktree:fixture",
        "assignment_revision": 3,
        "fencing_epoch": 2,
    }


@pytest.fixture(scope="module")
def measured_result(
    tmp_path_factory: pytest.TempPathFactory,
) -> MeasuredResult:
    repository_path, cloned_idle = _prepared_measured_clone(
        tmp_path_factory.mktemp("idle-real-process")
    )
    repository = cloned_idle.repository_identity(repository_path)
    result = cloned_idle.run_benchmark(
        repository=repository_path,
        identities=_identities(repository),
        manifest_path=repository_path / OWNED_PATHS[0],
        matrix_path=repository_path / idle.MATRIX_RELATIVE_PATH,
    )
    return result, repository, repository_path


def _rehash(result: dict[str, Any]) -> dict[str, Any]:
    result["content_sha256"] = idle.result_content_sha256(result)
    return result


def test_manifest_binds_exact_frozen_matrix_and_scoped_nonclaims() -> None:
    manifest = idle.load_manifest(MANIFEST_PATH)

    assert idle.RESULT_SCHEMA == "casf/idle-benchmark@1"
    assert idle.validate_matrix_binding(manifest, MATRIX_PATH)["sha256"] == idle.MATRIX_SHA256
    assert manifest["state"] == "specification_only"
    assert manifest["authoritative"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["execution"] == {
        "measurement_scope": "single_supervisor_population_hermetic_real_process_probe",
        "child_role": "state_owner_event_wait_probe",
        "configured_supervisor_population": 1,
        "idle_duration_seconds": 1,
        "wake_timeout_seconds": 5,
        "wait_interface": "StateOwnerEventWait@1",
        "network_permitted": False,
        "direct_database_access_permitted": False,
        "ducklake_scheduling_authority_permitted": False,
    }
    assert "multi-supervisor" in " ".join(manifest["nonclaims"])
    assert "production" in " ".join(manifest["nonclaims"])
    assert "not externally signed" in " ".join(manifest["nonclaims"])
    assert manifest["activity_sensors"] == [dict(sensor) for sensor in idle.ACTIVITY_SENSORS]


def test_real_child_measures_idle_zeros_and_lossless_injected_wake(
    measured_result: MeasuredResult,
) -> None:
    result, repository, repository_path = measured_result

    replay = idle.validate_result(
        result,
        idle.load_manifest(MANIFEST_PATH),
        matrix_path=MATRIX_PATH,
        current_identity=repository,
    )
    process = replay["process_evidence"]
    assert process["child_pid"] != os.getpid()
    assert process["parent_pid"] == os.getpid()
    assert process["process_start_ticks"] > 0
    assert process["birth_verified_by_parent"] is True
    assert process["probe_processes_observed"] == 1
    assert process["boot_id"] == idle._boot_id()
    assert process["returncode"] == 0
    assert result["configured_supervisor_population"] == 1
    assert Path(result["source_evidence"]["repository_root"]) == repository_path

    idle_measurement = replay["idle_measurement"]
    assert idle_measurement["bounded_waits"] == 1
    assert idle_measurement["measured_duration_ns"] >= 975_000_000
    assert idle_measurement["event_source_queries"] == 1
    assert idle_measurement["metrics"] == dict.fromkeys(idle.REQUIRED_IDLE_ZERO_METRICS, 0)
    assert idle_measurement["activity_snapshot"]["sensor_catalog_sha256"] == (
        idle._activity_sensor_catalog_sha256()
    )
    assert idle_measurement["activity_snapshot"]["values"] == idle_measurement["metrics"]

    wake = replay["wake_verification"]
    assert wake["injected_events"] == wake["delivered_events"] == 1
    assert wake["event_source_queries"] == 2
    assert wake["wait_wakeups"] == 1
    assert wake["lost_deliveries"] == wake["duplicate_deliveries"] == 0
    assert wake["zero_work_metrics_after"] == dict.fromkeys(idle.REQUIRED_ZERO_WORK_METRICS, 0)
    assert wake["activity_snapshot"]["values"]["wakeup_count"] == 1
    assert wake["activity_snapshot"]["values"]["event_deliveries"] == 1


def test_parent_sensor_log_is_complete_replayable_and_process_bound(
    measured_result: MeasuredResult,
) -> None:
    result, _repository, _repository_path = measured_result
    evidence = result["sensor_evidence"]
    spans = evidence["spans"]

    assert [item["child_span"]["kind"] for item in spans] == list(idle.SPAN_KINDS)
    assert spans[0]["previous_sha256"] == idle.FIRST_SPAN_PARENT
    assert all(
        later["previous_sha256"] == earlier["span_sha256"]
        for earlier, later in zip(spans[:-1], spans[1:], strict=True)
    )
    assert evidence["terminal_sha256"] == spans[-1]["span_sha256"]
    assert all(
        item["child_span"]["child_pid"] == result["process_evidence"]["child_pid"] for item in spans
    )
    assert all(
        item["child_span"]["boot_id"] == result["process_evidence"]["boot_id"] for item in spans
    )
    assert result["content_sha256"] == idle.result_content_sha256(result)


def test_owner_local_measurement_does_not_fake_typed_quack_live_availability(
    measured_result: MeasuredResult,
) -> None:
    result, _repository, _repository_path = measured_result

    assert result["wait_capability"]["qualification"] == "owner_local_hermetic_only"
    assert result["wait_capability"]["remote_quack_transport_qualified"] is False
    assert result["typed_quack_live"] == idle.TYPED_QUACK_UNAVAILABLE
    assert "metrics" not in result["typed_quack_live"]
    assert result["authoritative"] is False
    assert result["promotion_eligible"] is False


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda value: value.update({"measurement_scope": "typed_quack_live"}),
            "scope|state",
        ),
        (
            lambda value: value["typed_quack_live"].update(
                {"availability": "available", "execution_status": "passed"}
            ),
            "scope|state",
        ),
        (
            lambda value: value["process_evidence"].update(
                {"child_pid": value["process_evidence"]["child_pid"] + 1}
            ),
            "false birth|process identity",
        ),
        (
            lambda value: value["sensor_evidence"]["spans"].pop(2),
            "missing a required process span",
        ),
        (
            lambda value: value["sensor_evidence"].update({"terminal_sha256": "0" * 64}),
            "terminal hash",
        ),
    ],
)
def test_fake_live_missing_sensor_and_false_birth_fail_closed(
    measured_result: MeasuredResult,
    mutation: Any,
    match: str,
) -> None:
    candidate = deepcopy(measured_result[0])
    mutation(candidate)
    _rehash(candidate)

    with pytest.raises(idle.IdleBenchmarkError, match=match):
        idle.validate_result(
            candidate, matrix_path=MATRIX_PATH, current_identity=measured_result[1]
        )


def test_tampered_measured_zero_cannot_survive_sensor_replay(
    measured_result: MeasuredResult,
) -> None:
    candidate = deepcopy(measured_result[0])
    candidate["sensor_evidence"]["spans"][2]["child_span"]["payload"]["activity_snapshot_after"][
        "values"
    ]["task_board_scans"] = 1
    _rehash(candidate)

    with pytest.raises(idle.IdleBenchmarkError, match="measured zero|raw sensor hash"):
        idle.validate_result(
            candidate, matrix_path=MATRIX_PATH, current_identity=measured_result[1]
        )


def test_missing_or_changed_activity_sensor_provenance_fails_closed(
    measured_result: MeasuredResult,
) -> None:
    missing = deepcopy(measured_result[0])
    missing["activity_sensors"].pop()
    _rehash(missing)
    with pytest.raises(idle.IdleBenchmarkError, match="missing a required sensor"):
        idle.validate_result(missing, matrix_path=MATRIX_PATH, current_identity=measured_result[1])

    changed = deepcopy(measured_result[0])
    changed["sensor_evidence"]["spans"][1]["child_span"]["payload"]["activity_snapshot_before"][
        "sensor_catalog_sha256"
    ] = "0" * 64
    _rehash(changed)
    with pytest.raises(idle.IdleBenchmarkError, match="sensor provenance|raw sensor hash"):
        idle.validate_result(changed, matrix_path=MATRIX_PATH, current_identity=measured_result[1])


def test_import_shadow_origin_and_primary_source_hash_fail_closed(
    measured_result: MeasuredResult,
) -> None:
    shadowed = deepcopy(measured_result[0])
    module_name = next(iter(idle.EXPECTED_MODULE_ORIGINS))
    shadowed["process_evidence"]["imported_module_origins"][module_name] = "/tmp/shadowed-module.py"
    _rehash(shadowed)
    with pytest.raises(idle.IdleBenchmarkError, match="origins differ"):
        idle.validate_result(shadowed, matrix_path=MATRIX_PATH, current_identity=measured_result[1])

    changed_source = deepcopy(measured_result[0])
    source_path = idle.SOURCE_RELATIVE_PATHS[-1]
    changed_source["source_evidence"]["source_sha256"][source_path] = "0" * 64
    _rehash(changed_source)
    with pytest.raises(idle.IdleBenchmarkError, match="source hashes are stale"):
        idle.validate_result(
            changed_source, matrix_path=MATRIX_PATH, current_identity=measured_result[1]
        )


def test_replay_requires_exact_current_commit_and_tree(measured_result: MeasuredResult) -> None:
    with pytest.raises(idle.IdleBenchmarkError, match="current identity is required"):
        idle.validate_result(measured_result[0], matrix_path=MATRIX_PATH)
    with pytest.raises(idle.IdleBenchmarkError, match="closed schema"):
        idle.validate_result(measured_result[0], matrix_path=MATRIX_PATH, current_identity={})


def test_stale_tree_and_changed_matrix_fail_closed(
    measured_result: MeasuredResult, tmp_path: Path
) -> None:
    result, repository, _repository_path = measured_result
    with pytest.raises(idle.IdleBenchmarkError, match="stale"):
        idle.validate_result(
            result,
            matrix_path=MATRIX_PATH,
            current_identity={
                "repository_commit": repository["repository_commit"],
                "repository_tree": "0" * 40,
            },
        )

    stale_matrix = tmp_path / "matrix.yaml"
    stale_matrix.write_text(
        MATRIX_PATH.read_text(encoding="utf-8") + "# changed\n", encoding="utf-8"
    )
    with pytest.raises(idle.IdleBenchmarkError, match="matrix content is stale"):
        idle.validate_result(result, matrix_path=stale_matrix, current_identity=repository)


def test_dirty_repository_is_rejected_before_a_child_can_claim_current_tree(tmp_path: Path) -> None:
    repository_path = tmp_path / "repository"
    _clean_repository(repository_path)
    (repository_path / "uncommitted.txt").write_text("dirty\n", encoding="utf-8")

    with pytest.raises(idle.IdleBenchmarkError, match="dirty"):
        idle.repository_identity(repository_path)


def test_unrelated_clean_repository_cannot_label_measured_source(tmp_path: Path) -> None:
    repository_path = tmp_path / "repository"
    repository = _clean_repository(repository_path)

    with pytest.raises(idle.IdleBenchmarkError, match="exact repository containing"):
        idle.run_benchmark(
            repository=repository_path,
            identities=_identities(repository),
            manifest_path=MANIFEST_PATH,
            matrix_path=MATRIX_PATH,
        )


def test_repository_change_during_child_run_invalidates_evidence(tmp_path: Path) -> None:
    repository_path, cloned_idle = _prepared_measured_clone(tmp_path)
    repository = cloned_idle.repository_identity(repository_path)
    concurrent_file = repository_path / "concurrent-change.txt"

    def change_tree_during_wait() -> None:
        time.sleep(0.6)
        concurrent_file.write_text("changed during measurement\n", encoding="utf-8")

    writer = threading.Thread(target=change_tree_during_wait)
    writer.start()
    try:
        with pytest.raises(cloned_idle.IdleBenchmarkError, match="dirty|changed during the run"):
            cloned_idle.run_benchmark(
                repository=repository_path,
                identities=_identities(repository),
                manifest_path=repository_path / OWNED_PATHS[0],
                matrix_path=repository_path / idle.MATRIX_RELATIVE_PATH,
            )
    finally:
        writer.join()
        concurrent_file.unlink(missing_ok=True)


def test_duplicate_keys_and_unknown_fields_are_rejected(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema": 1, "schema": 2}', encoding="utf-8")
    with pytest.raises(idle.IdleBenchmarkError, match="duplicate"):
        idle._read_object(duplicate)

    manifest = idle.load_manifest(MANIFEST_PATH)
    manifest["live"] = True
    with pytest.raises(idle.IdleBenchmarkError, match="closed schema"):
        idle.validate_manifest(manifest)


def test_caller_authored_old_live_observation_is_not_an_admissible_identity() -> None:
    old_observation = {
        "measurement_mode": "live",
        "metrics": dict.fromkeys(idle.REQUIRED_IDLE_ZERO_METRICS, 0),
    }

    with pytest.raises(idle.IdleBenchmarkError, match="closed schema"):
        idle._validate_identities(old_observation)


def test_cli_requires_repository_and_closed_identities(capsys: pytest.CaptureFixture[str]) -> None:
    assert idle.main(["--manifest", str(MANIFEST_PATH), "--matrix", str(MATRIX_PATH)]) == 1
    failure = json.loads(capsys.readouterr().out)
    assert failure["execution_status"] == "invalid"
    assert "required" in failure["error"]
