"""Closed contracts for the CASF event-driven idle benchmark recipe."""

from __future__ import annotations

import importlib.util
import json
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
RUNNER_PATH = ROOT / "benchmarks/agent_supervisor/causal_event_federation/run_idle.py"
MANIFEST_PATH = ROOT / "benchmarks/agent_supervisor/causal_event_federation/idle_manifest.json"
SPEC = importlib.util.spec_from_file_location("casf_idle_benchmark", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
idle = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(idle)


def _observation() -> dict[str, object]:
    manifest = idle.load_manifest(MANIFEST_PATH)
    started = datetime(2030, 1, 1, tzinfo=UTC)
    finished = started + timedelta(seconds=60)
    return {
        "schema": idle.OBSERVATION_SCHEMA,
        "benchmark_id": idle.BENCHMARK_ID,
        "manifest_sha256": idle.manifest_sha256(manifest),
        "run_id": "idle-run:test",
        "started_at": started.isoformat().replace("+00:00", "Z"),
        "finished_at": finished.isoformat().replace("+00:00", "Z"),
        "measurement_mode": "live",
        "identities": {
            "repository_commit": "commit:test",
            "repository_tree": "tree:test",
            "control_plane_generation": 7,
            "schema_fingerprint": "schema:test",
            "policy_ref": "policy:test",
            "policy_revision": "policy-revision:test",
            "capability_ref": "capability:test",
            "federation_id": "federation:test",
            "supervisor_id": "supervisor:test",
            "task_id": "task:test",
            "attempt_id": "attempt:test",
            "worktree_id": "worktree:test",
            "assignment_revision": 3,
            "fencing_epoch": 2,
        },
        "wait_capability": {
            "available": True,
            "interface": "TypedStateOwnerEventWait@1",
            "client_interface": "QuackStateClientEventWait@1",
            "transport": "typed_state_owner_bounded_long_wait",
            "server_owned": True,
            "blocking_condition": True,
            "adaptive_polling": False,
            "event_driven_qualified": True,
            "idle_repeated_database_scans": False,
        },
        "activity": {
            "blocked_server_owned_event_wait": 1,
            "required_lease_heartbeat": 0,
            "bounded_health_deadline": 0,
            "explicit_recovery_timer": 0,
        },
        "metrics": {
            "task_board_scans": 0,
            "model_calls": 0,
            "context_recompilations": 0,
            "unchanged_state_writes": 0,
            "wakeup_count": 0,
            "event_deliveries": 0,
            "duplicate_committed_effects": 0,
            "lost_deliveries": 0,
            "completed_waits": 1,
            "server_wait_queries": 1,
        },
        "measurement_source": "typed-state-owner-runtime@1",
        "simulation": False,
        "direct_database_access": False,
        "network_used": False,
        "ducklake_scheduling_authority": False,
        "promotion_eligible": False,
    }


def test_manifest_is_frozen_non_authoritative_idle_contract() -> None:
    manifest = idle.load_manifest(MANIFEST_PATH)

    assert manifest["state"] == "specification_only"
    assert manifest["authoritative"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["required_zero_metrics"] == list(idle.REQUIRED_ZERO_METRICS)
    assert "blocked_server_owned_event_wait" in manifest["permitted_activity"]


def test_valid_live_observation_is_identity_bound_and_non_promoting() -> None:
    report = idle.validate_observation(
        _observation(),
        current_identity={"repository_commit": "commit:test", "repository_tree": "tree:test"},
    )

    assert report["verified"] is True
    assert report["authoritative"] is False
    assert report["promotion_eligible"] is False
    assert report["metrics"]["task_board_scans"] == 0


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("metrics", "task_board_scans"), 1, "task_board_scans"),
        (("metrics", "model_calls"), 1, "model_calls"),
        (("metrics", "unchanged_state_writes"), 1, "unchanged_state_writes"),
        (("wait_capability", "adaptive_polling"), True, "qualified typed wait"),
        (("wait_capability", "event_driven_qualified"), False, "qualified typed wait"),
        (("measurement_mode",), "hermetic", "live typed-state-owner"),
        (("simulation",), True, "prohibited capability"),
        (("ducklake_scheduling_authority",), True, "prohibited capability"),
    ],
)
def test_forbidden_idle_activity_or_unqualified_capability_fails_closed(
    path: tuple[str, ...], value: object, match: str
) -> None:
    observation = deepcopy(_observation())
    target: dict[str, object] = observation
    for key in path[:-1]:
        target = target[key]  # type: ignore[assignment,index]
    target[path[-1]] = value

    with pytest.raises(idle.IdleBenchmarkError, match=match):
        idle.validate_observation(observation)


def test_missing_identity_stale_tree_and_short_window_fail_closed() -> None:
    missing = deepcopy(_observation())
    del missing["identities"]["fencing_epoch"]  # type: ignore[index]
    with pytest.raises(idle.IdleBenchmarkError, match="closed schema"):
        idle.validate_observation(missing)

    stale = _observation()
    with pytest.raises(idle.IdleBenchmarkError, match="stale"):
        idle.validate_observation(stale, current_identity={"repository_tree": "tree:other"})

    short = _observation()
    short["finished_at"] = "2030-01-01T00:00:59Z"
    with pytest.raises(idle.IdleBenchmarkError, match="shorter"):
        idle.validate_observation(short)


def test_duplicate_json_keys_and_unknown_evidence_fields_are_rejected(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema": 1, "schema": 2}', encoding="utf-8")
    with pytest.raises(idle.IdleBenchmarkError, match="duplicate"):
        idle._read_object(duplicate)

    observation = _observation()
    observation["sql"] = "SELECT * FROM control"
    with pytest.raises(idle.IdleBenchmarkError, match="closed schema"):
        idle.validate_observation(observation)


def test_cli_refuses_invalid_observation_without_emitting_a_passing_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    observation = _observation()
    observation["metrics"]["event_deliveries"] = 1  # type: ignore[index]
    path = tmp_path / "observation.json"
    path.write_text(json.dumps(observation), encoding="utf-8")

    assert idle.main(["--observation", str(path), "--manifest", str(MANIFEST_PATH)]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["verified"] is False
