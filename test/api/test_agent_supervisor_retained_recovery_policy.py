"""Hermetic recovery-denial tests: authored metadata and private files only."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest


RESCUE = Path(__file__).resolve().parents[2] / "ipfs_accelerate_py/agent_supervisor/rescue"


@pytest.fixture
def isolated(monkeypatch):
    # Load the actual two owners without importing the product package or any
    # default home inventory, provider, database, or root conftest helper.
    package = ModuleType("retained_recovery_test")
    package.__path__ = [str(RESCUE)]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    probe = ModuleType(package.__name__ + ".live_board_probe")
    probe.COMPLETED = frozenset({"completed", "done", "accepted", "succeeded", "complete"})

    def no_population_pin(board):
        if "native_population" in board:
            raise ValueError("this private fixture does not admit a population pin")
        return None

    probe.native_population = no_population_pin

    def read_private_json(path):
        if not path.exists():
            return {}
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise ValueError("object required")
        return value

    probe.read_json_object = read_private_json
    monkeypatch.setitem(sys.modules, probe.__name__, probe)
    loaded = []
    for name in ("recovery_policy", "fleet_watchdog"):
        spec = importlib.util.spec_from_file_location(
            package.__name__ + "." + name, RESCUE / (name + ".py"))
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
        loaded.append(module)
    return tuple(loaded)


def policy(**changes):
    return {"schema": "agent-supervisor/owner-retention@1", "mode": "retain",
            "profile_id": "fixture-m70", "reason": "retain the admitted owner", **changes}


def observation(**changes):
    return {"board_id": "private", "health": "healthy", "reason_codes": [],
            "busy": False, "complete": False, "progress_token": "fixture", **changes}


@pytest.mark.parametrize("marker", [
    {"selection_idle_reason": "expired_attempt_settlement_unavailable"},
    {"reason": "production_attempt_settlement_required"},
])
@pytest.mark.parametrize("shape", [
    "top", "details", "lane_state", "lane_summary", "lane_map", "nested_cycle",
])
def test_retained_markers_withhold_at_each_observation_boundary(isolated, marker, shape):
    helper, _ = isolated
    if shape == "top":
        sample = dict(marker)
    elif shape == "details":
        sample = {"details": dict(marker)}
    elif shape == "lane_state":
        sample = {"details": {"lanes": [{"state": dict(marker)}]}}
    elif shape == "lane_summary":
        sample = {"lanes": [{"summary": dict(marker)}]}
    elif shape == "lane_map":
        sample = {"lane_states": {"lane-3": {"summary": dict(marker)}}}
    else:
        sample = {"details": {"lanes": [{"state": {"last_cycle_result": dict(marker)}}]}}
    before = copy.deepcopy(sample)
    assert helper.retained_attempt_wait(sample) is True
    assert helper.recovery_denial({}, sample) == "retained_attempt_settlement_wait"
    assert sample == before


def test_unknown_typed_custody_and_required_boolean_are_denials_not_authority(isolated):
    helper, _ = isolated
    for sample in (
        {"retained_attempt_settlement_required": True},
        {"details": {"custody_observation": {
            "schema": "ipfs_accelerate_py/attempt-custody-observation@1",
            "callback_outcome": "unknown", "completion_authority": False,
            "retry_authorized": False, "settlement_authority": False}}},
    ):
        before = copy.deepcopy(sample)
        assert helper.retained_attempt_wait(sample) is True
        assert helper.recovery_denial({}, sample) == "retained_attempt_settlement_wait"
        assert sample == before
    for sample in (
        {"retained_attempt_settlement_required": 1},
        {"schema": "foreign", "callback_outcome": "unknown"},
        {"payload": {"reason": "production_attempt_settlement_required"}},
        {"summary": "expired_attempt_settlement_unavailable"},
    ):
        assert helper.retained_attempt_wait(sample) is False


def test_retained_scan_is_cycle_safe_and_large_metadata_only_withholds(isolated):
    helper, _ = isolated
    sample = {}
    sample["state"] = sample
    assert helper.retained_attempt_wait(sample) is False
    sample["summary"] = {"reason": "production_attempt_settlement_required"}
    assert helper.retained_attempt_wait(sample) is True
    assert helper.retained_attempt_wait({"lanes": [{} for _ in range(4097)]}) is True


@pytest.mark.parametrize("value", [
    None, False, "retain", {},
    policy(schema="foreign"), policy(mode="restart"), policy(extra=True),
    policy(profile_id=""), policy(profile_id=" "), policy(profile_id="x" * 257),
    policy(profile_id=True), policy(reason=""), policy(reason="\t"),
    policy(reason="x" * 1025), policy(reason=1),
])
def test_every_malformed_present_policy_retains_owner_and_denies(isolated, value):
    helper, _ = isolated
    config = {"recovery_policy": value}
    before = copy.deepcopy(config)
    assert helper.owner_retention_required(config) is True
    assert helper.recovery_denial(config, {}) == "invalid_owner_retention_policy"
    assert config == before


def test_policy_is_closed_bounded_and_absence_preserves_ordinary_behavior(isolated):
    helper, _ = isolated
    assert helper.owner_retention_required({}) is False
    assert helper.recovery_denial({}, {"health": "stopped"}) == ""
    config = {"recovery_policy": policy(profile_id="p" * 256, reason="r" * 1024)}
    assert helper.owner_retention_required(config) is True
    assert helper.recovery_denial(config, {}) == "native_owner_retention_required"
    marker = {"reason": "production_attempt_settlement_required"}
    assert helper.recovery_denial(config, marker) == "retained_attempt_settlement_wait"
    config["recovery_policy"] = None
    assert helper.recovery_denial(config, marker) == "invalid_owner_retention_policy"


@pytest.mark.parametrize("health,reasons", [
    ("stalled", ["source_integrity_not_verified", "no_task_progress"]),
    ("stopped", ["owner_not_ready"]),
    ("complete", []),
    ("operator_hold", []),
])
def test_retained_wait_precedes_generic_classification_and_complete(isolated, health, reasons):
    _, fleet = isolated
    sample = observation(health=health, reason_codes=reasons, complete=health == "complete",
        details={"selection_idle_reason": "expired_attempt_settlement_unavailable",
                 "task_counts": {"retrying": 1, "todo": 2}, "owner_ready": True,
                 "lanes": [{"supervisor": {}, "daemon": {}}]})
    assert fleet.classify_stall(sample) == "retained_attempt_settlement_wait"
    assert "retained_attempt_settlement_wait" in fleet.WAIT_STALLS
    board = {"id": "private", "publication": {"manifest": "must-not-publish"}}
    assert fleet.select_action({"health": health, "observation": sample,
                               "stall_class": "stalled_no_progress"}, board, 100) == ""


@pytest.mark.parametrize("denial", ["retained", "policy", "invalid_policy"])
@pytest.mark.parametrize("health", ["healthy", "stopped", "complete"])
def test_applied_tick_denial_preserves_recovery_custody_before_all_effects(
        isolated, tmp_path, monkeypatch, denial, health):
    _, fleet = isolated
    board = {"id": "private", "cwd": str(tmp_path), "probe": {"argv": ["private-probe"]},
             "ensure": {"argv": ["must-not-ensure"]},
             "repair": {"argv": ["must-not-enqueue"]},
             "publication": {"manifest": "must-not-publish"}, "stall_seconds": 0}
    sample = observation(health=health, complete=health == "complete")
    if denial == "retained":
        sample["details"] = {"lanes": [{"summary": {
            "reason": "production_attempt_settlement_required"}}]}
        expected = "retained_attempt_settlement_wait"
    else:
        board["recovery_policy"] = policy() if denial == "policy" else None
        expected = ("native_owner_retention_required" if denial == "policy"
                    else "invalid_owner_retention_policy")
    hold = tmp_path / "native-owner.HOLD"
    hold.write_text("operator-owned retained custody")
    board["hold_files"] = [str(hold)]
    root = tmp_path / "watch"
    recovery = {"attempts": 9, "ensure_attempts": 4, "ensure_owner_epoch": "old-owner",
                "pending_action": "repair", "next_action_at": 7777,
                "last_action": "ensure", "last_action_at": 12,
                "last_action_result": {"status": "unresolved"}}
    fleet.write_json(root / "private/state.json", {
        **recovery, "health": "stalled", "incident_since": 1, "last_progress_at": 1})
    job = root / "repairs/private/job.json"
    fleet.write_json(job, {"status": "running", "attempts": 3, "next_attempt_at": 9999})
    job_before, hold_before = job.read_bytes(), hold.read_bytes()
    calls = []

    def probe_only(spec, **kwargs):
        calls.append(spec)
        assert spec == board["probe"]
        return {"returncode": 0, "stdout": json.dumps(sample), "stderr": "", "timed_out": False}

    def forbidden(*args, **kwargs):
        pytest.fail("a denial cannot stop, heal, publish, review holds or enqueue")

    monkeypatch.setattr(fleet, "command", forbidden)
    monkeypatch.setattr(Path, "home", forbidden)
    monkeypatch.setattr(fleet, "_ready_owner_epoch", lambda _: "new-owner")
    for module_name, function in (("fleet_holds", "review_board_holds"),
                                  ("fleet_completion", "publish_completed_board"),
                                  ("fleet_heals", "apply_supervisor_heal")):
        stub = ModuleType("retained_recovery_test." + module_name)
        setattr(stub, function, forbidden)
        monkeypatch.setitem(sys.modules, stub.__name__, stub)
    result = fleet.tick_board(board, root, apply=True, runner=probe_only, now=100)
    assert calls == [board["probe"]]
    assert result["planned_action"] == "" and result["recovery_denial"] == expected
    assert {key: result[key] for key in recovery} == recovery
    assert job.read_bytes() == job_before and hold.read_bytes() == hold_before
    assert not (root / "private/incident.json").exists()
    assert not (root / "private/last_action.json").exists()
    assert fleet.read_json(root / "private/state.json") == result


def test_policy_clear_does_not_leave_a_stale_denial_and_ordinary_selection_survives(isolated, tmp_path):
    _, fleet = isolated
    board = {"id": "private", "cwd": str(tmp_path), "probe": {"argv": ["private-probe"]}}
    sample = observation(health="unknown")
    state = {"health": "unknown", "observation": sample, "stall_class": "probe_failed",
             "incident_since": 0, "next_action_at": 0}
    assert fleet.select_action(state, board, 100) == "repair"
    root = tmp_path / "watch"
    fleet.write_json(root / "private/state.json", {"health": "healthy",
        "last_progress_at": 99, "recovery_denial": "native_owner_retention_required"})
    result = fleet.tick_board(board, root, apply=False, now=100,
        runner=lambda *a, **k: {"returncode": 0, "stdout": json.dumps(observation()),
                              "stderr": "", "timed_out": False})
    assert "recovery_denial" not in result
    assert result["planned_action"] == ""
