"""Closed, fail-closed contracts for the CASF twelve-supervisor benchmark."""

from __future__ import annotations

import importlib.util
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
RUNNER_PATH = ROOT / "benchmarks/agent_supervisor/causal_event_federation/run_parallel.py"
MANIFEST_PATH = ROOT / "benchmarks/agent_supervisor/causal_event_federation/parallel_manifest.json"
MATRIX_PATH = ROOT / "benchmarks/agent_supervisor/causal_event_federation/matrix.yaml"
SPEC = importlib.util.spec_from_file_location("casf_parallel_benchmark", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
parallel = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(parallel)


def test_manifest_freezes_twelve_real_processes_and_comparison_assurance() -> None:
    manifest = parallel.load_manifest(MANIFEST_PATH)

    assert parallel.RESULT_SCHEMA == "casf/parallel-benchmark@1"
    assert parallel.validate_matrix_binding(manifest, MATRIX_PATH)["sha256"] == parallel.MATRIX_SHA256
    assert manifest["state"] == "capability_blocked_specification_only"
    assert manifest["authoritative"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["execution"] == {
        "measurement_scope": "twelve_independent_supervisor_processes_qualified_live_profile",
        "required_supervisor_processes": 12,
        "subprocess_budget": 12,
        "real_independent_processes_required": True,
        "in_process_simulation_qualifies": False,
        "network_permitted": False,
        "direct_database_access_permitted": False,
        "ducklake_scheduling_authority_permitted": False,
        "launch_permitted": False,
    }
    assert manifest["parallel_comparison"] == {
        "baseline_qualified_supervisors": 1,
        "comparison_identity": "same-host-tasks-providers-tests-proofs-budgets",
        "minimum_accepted_task_throughput_multiplier": 3.0,
        "lower_assurance_permitted": False,
    }
    assert tuple(manifest["zero_tolerance_gates"]) == parallel.ZERO_TOLERANCE_GATES
    assert "simulation" in " ".join(manifest["nonclaims"])
    assert "production" in " ".join(manifest["nonclaims"])


def test_missing_live_capacity_fails_before_any_child_or_repository_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("blocked benchmark must not launch or probe")

    monkeypatch.setattr(parallel.subprocess, "Popen", forbidden)
    monkeypatch.setattr(parallel, "repository_identity", forbidden)

    with pytest.raises(parallel.ParallelCapabilityUnavailable, match="live capability is unavailable"):
        parallel.run_benchmark(repository=ROOT, identities={})


def test_live_capability_descriptor_is_explicitly_unavailable_not_a_receipt() -> None:
    capability = parallel.live_capability(parallel.load_manifest(MANIFEST_PATH))

    assert capability == parallel.LIVE_CAPABILITY_UNAVAILABLE
    assert capability["availability"] == "unavailable"
    assert capability["execution_status"] == "not_run"
    assert capability["metrics_omitted"] is True
    assert "result" not in capability
    assert "receipt" not in capability


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda value: value["execution"].update({"launch_permitted": True}),
            "may not launch or simulate",
        ),
        (
            lambda value: value["execution"].update({"in_process_simulation_qualifies": True}),
            "may not launch or simulate",
        ),
        (
            lambda value: value["parallel_comparison"].update({"lower_assurance_permitted": True}),
            "assurance or throughput",
        ),
        (
            lambda value: value["live_capability"].update(
                {"availability": "available", "execution_status": "passed"}
            ),
            "must remain unavailable/not-run",
        ),
        (
            lambda value: value["zero_tolerance_gates"].pop(),
            "zero-tolerance safety gates",
        ),
    ],
)
def test_manifest_weakening_and_fake_live_status_fail_closed(
    mutation: Any, match: str
) -> None:
    candidate = deepcopy(parallel.load_manifest(MANIFEST_PATH))
    mutation(candidate)

    with pytest.raises(parallel.ParallelBenchmarkError, match=match):
        parallel.validate_manifest(candidate)


def test_alleged_result_is_rejected_even_if_content_address_is_self_consistent() -> None:
    alleged = {
        "schema": parallel.RESULT_SCHEMA,
        "benchmark_id": parallel.BENCHMARK_ID,
        "execution_status": "measured",
        "availability": "available",
        "authoritative": False,
        "promotion_eligible": False,
        "content_sha256": "",
    }
    alleged["content_sha256"] = parallel.result_content_sha256(alleged)

    with pytest.raises(parallel.ParallelCapabilityUnavailable, match="live capability is unavailable"):
        parallel.validate_result(alleged, matrix_path=MATRIX_PATH, current_identity={})


def test_changed_matrix_duplicate_json_and_unknown_manifest_fields_fail_closed(
    tmp_path: Path,
) -> None:
    stale_matrix = tmp_path / "matrix.yaml"
    stale_matrix.write_text(MATRIX_PATH.read_text(encoding="utf-8") + "# changed\n", encoding="utf-8")
    with pytest.raises(parallel.ParallelBenchmarkError, match="matrix content is stale"):
        parallel.validate_matrix_binding(parallel.load_manifest(MANIFEST_PATH), stale_matrix)

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema": 1, "schema": 2}', encoding="utf-8")
    with pytest.raises(parallel.ParallelBenchmarkError, match="duplicate"):
        parallel._read_object(duplicate)

    candidate = parallel.load_manifest(MANIFEST_PATH)
    candidate["fake_live_observation"] = {"twelve_processes": 12}
    with pytest.raises(parallel.ParallelBenchmarkError, match="closed schema"):
        parallel.validate_manifest(candidate)


def test_cli_requires_closed_repository_and_identity_arguments(capsys: pytest.CaptureFixture[str]) -> None:
    assert parallel.main(["--manifest", str(MANIFEST_PATH), "--matrix", str(MATRIX_PATH)]) == 1
    failure = json.loads(capsys.readouterr().out)
    assert failure["schema"] == parallel.RESULT_SCHEMA
    assert failure["execution_status"] == "not_run"
    assert failure["availability"] == "unavailable"
    assert "required" in failure["error"]


def test_closed_identity_contract_rejects_stale_or_unrelated_inputs() -> None:
    with pytest.raises(parallel.ParallelBenchmarkError, match="closed schema"):
        parallel._validate_identities({})

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
        "task_id": "CASF-038",
        "attempt_id": "attempt:fixture",
        "worktree_id": "worktree:fixture",
        "assignment_revision": 1,
        "fencing_epoch": 1,
    }
    with pytest.raises(parallel.ParallelBenchmarkError, match="CASF-039"):
        parallel._validate_identities(identities)
