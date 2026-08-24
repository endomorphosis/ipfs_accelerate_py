"""Closed, fail-closed contracts for the CASF 256-agent load benchmark."""

from __future__ import annotations

import importlib.util
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
RUNNER_PATH = ROOT / "benchmarks/agent_supervisor/causal_event_federation/run_load.py"
MANIFEST_PATH = ROOT / "benchmarks/agent_supervisor/causal_event_federation/load_manifest.json"
MATRIX_PATH = ROOT / "benchmarks/agent_supervisor/causal_event_federation/matrix.yaml"
SPEC = importlib.util.spec_from_file_location("casf_load_benchmark", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
load = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(load)


def test_manifest_freezes_256_agent_bounded_load_and_safety_contract() -> None:
    manifest = load.load_manifest(MANIFEST_PATH)

    assert load.RESULT_SCHEMA == "casf/load-benchmark@1"
    assert load.validate_matrix_binding(manifest, MATRIX_PATH)["sha256"] == load.MATRIX_SHA256
    assert manifest["state"] == "capability_blocked_specification_only"
    assert manifest["authoritative"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["execution"] == {
        "measurement_scope": (
            "twelve_independent_supervisor_processes_256_registered_agents_qualified_live_profile"
        ),
        "required_supervisor_processes": 12,
        "registered_logical_agents": 256,
        "maximum_concurrent_subagents": 64,
        "minimum_bounded_tasks": 1000,
        "minimum_event_deliveries_with_replay": 100000,
        "subprocess_budget": 12,
        "real_independent_processes_required": True,
        "in_process_simulation_qualifies": False,
        "network_permitted": False,
        "direct_database_access_permitted": False,
        "ducklake_scheduling_authority_permitted": False,
        "launch_permitted": False,
    }
    assert tuple(manifest["zero_tolerance_gates"]) == load.ZERO_TOLERANCE_GATES
    assert "simulation" in " ".join(manifest["nonclaims"])
    assert "completion" in " ".join(manifest["nonclaims"])


def test_missing_live_capacity_fails_before_child_or_repository_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("blocked benchmark must not launch or probe")

    monkeypatch.setattr(load.subprocess, "Popen", forbidden)
    monkeypatch.setattr(load, "repository_identity", forbidden)

    with pytest.raises(load.LoadCapabilityUnavailable, match="live capability is unavailable"):
        load.run_benchmark(repository=ROOT, identities={})


def test_live_capability_descriptor_is_explicitly_unavailable_not_a_receipt() -> None:
    capability = load.live_capability(load.load_manifest(MANIFEST_PATH))

    assert capability == load.LIVE_CAPABILITY_UNAVAILABLE
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
            lambda value: value["execution"].update({"registered_logical_agents": 257}),
            "may not launch or simulate",
        ),
        (
            lambda value: value["execution"].update({"in_process_simulation_qualifies": True}),
            "may not launch or simulate",
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
    candidate = deepcopy(load.load_manifest(MANIFEST_PATH))
    mutation(candidate)

    with pytest.raises(load.LoadBenchmarkError, match=match):
        load.validate_manifest(candidate)


def test_alleged_result_is_rejected_even_if_content_address_is_self_consistent() -> None:
    alleged = {
        "schema": load.RESULT_SCHEMA,
        "benchmark_id": load.BENCHMARK_ID,
        "execution_status": "measured",
        "availability": "available",
        "authoritative": False,
        "promotion_eligible": False,
        "content_sha256": "",
    }
    alleged["content_sha256"] = load.result_content_sha256(alleged)

    with pytest.raises(load.LoadCapabilityUnavailable, match="live capability is unavailable"):
        load.validate_result(alleged, matrix_path=MATRIX_PATH, current_identity={})


def test_changed_matrix_duplicate_json_and_unknown_manifest_fields_fail_closed(tmp_path: Path) -> None:
    stale_matrix = tmp_path / "matrix.yaml"
    stale_matrix.write_text(MATRIX_PATH.read_text(encoding="utf-8") + "# changed\n", encoding="utf-8")
    with pytest.raises(load.LoadBenchmarkError, match="matrix content is stale"):
        load.validate_matrix_binding(load.load_manifest(MANIFEST_PATH), stale_matrix)

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema": 1, "schema": 2}', encoding="utf-8")
    with pytest.raises(load.LoadBenchmarkError, match="duplicate"):
        load._read_object(duplicate)

    candidate = load.load_manifest(MANIFEST_PATH)
    candidate["fake_live_observation"] = {"registered_agents": 256}
    with pytest.raises(load.LoadBenchmarkError, match="closed schema"):
        load.validate_manifest(candidate)


def test_cli_requires_closed_repository_and_identity_arguments(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert load.main(["--manifest", str(MANIFEST_PATH), "--matrix", str(MATRIX_PATH)]) == 1
    failure = json.loads(capsys.readouterr().out)
    assert failure["schema"] == load.RESULT_SCHEMA
    assert failure["execution_status"] == "not_run"
    assert failure["availability"] == "unavailable"
    assert "required" in failure["error"]


def test_closed_identity_contract_rejects_stale_or_unrelated_inputs() -> None:
    with pytest.raises(load.LoadBenchmarkError, match="closed schema"):
        load._validate_identities({})

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
        "task_id": "CASF-039",
        "attempt_id": "attempt:fixture",
        "worktree_id": "worktree:fixture",
        "assignment_revision": 1,
        "fencing_epoch": 1,
    }
    with pytest.raises(load.LoadBenchmarkError, match="CASF-040"):
        load._validate_identities(identities)
