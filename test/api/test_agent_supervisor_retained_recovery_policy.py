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


# Both original closed policy families share one validator. These literal
# controls keep coding diagnosis distinct from native-owner recovery.
def _interop_infrastructure_board(tmp_path):
    return {"id": "private", "cwd": str(tmp_path), "hold_files": [],
        "probe": {"argv": ["private-probe"]}, "repair": {"argv": ["private-repair"]},
        "probe_profile": "native_catalogue", "max_repair_attempts": 3,
        "failure_grace_seconds": 0, "cooldown_seconds": 180,
        "native_catalogue_binding": {"board_namespace": "private-native", "plan_root_cid": "plan-1",
            "repository_tree_id": "tree-1", "population_sha256": "a" * 64, "max_age_seconds": 180},
        "recovery_policy": {"profile": "infrastructure-only/v1",
            "repairable_stalls": ["native_catalogue_read_unavailable"],
            "max_actions_per_window": 1, "window_seconds": 86400, "circuit_breaker_seconds": 3600}}


def _interop_catalogue_failure(now=100):
    return observation(health="unknown", reason_codes=["native_catalogue_read_unavailable"],
        details={"native_catalogue": {"schema": "native-catalogue-diagnostic/v1",
            "board_namespace": "private-native", "plan_root_cid": "plan-1",
            "repository_tree_id": "tree-1", "population_sha256": "a" * 64,
            "completion_authority": False, "probe_observed_at_epoch": now, "fresh_read": False}})


def test_closed_infrastructure_policy_allows_only_explicit_coding_diagnosis(isolated, tmp_path):
    helper, fleet = isolated
    board = _interop_infrastructure_board(tmp_path)
    before = copy.deepcopy(board)
    assert helper.owner_retention_required(board) is True
    assert helper.recovery_denial(board, {}) == "infrastructure_only_recovery"
    assert helper.recovery_denial(board, {}, allow_infrastructure=True) == ""
    assert fleet.infrastructure_policy(board) == board["recovery_policy"]
    retained = {"recovery_policy": policy()}
    assert fleet.infrastructure_policy(retained) is None
    assert helper.recovery_denial(retained, {}, allow_infrastructure=True) == "native_owner_retention_required"
    assert board == before


@pytest.mark.parametrize("damage", ["mixed", "unknown", "missing_cap", "boolean_cap", "native_stall", "nonfinite", "optional_flag"])
def test_malformed_or_mixed_policy_cannot_enter_infrastructure_diagnosis(isolated, tmp_path, damage):
    helper, fleet = isolated
    board = _interop_infrastructure_board(tmp_path)
    p = board["recovery_policy"]
    if damage == "mixed": p.update(policy())
    elif damage == "unknown": p["reset_attempts"] = True
    elif damage == "missing_cap": board.pop("max_repair_attempts")
    elif damage == "boolean_cap": board["max_repair_attempts"] = True
    elif damage == "native_stall": p["repairable_stalls"] = ["native_catalogue_attempt_budget_exhausted"]
    elif damage == "nonfinite": p["window_seconds"] = float("nan")
    else: p["stop_own_repair_on_wait"] = 1
    assert helper.recovery_denial(board, {}, allow_infrastructure=True) == "invalid_owner_retention_policy"
    with pytest.raises(ValueError): fleet.infrastructure_policy(board)
    state = {"health": "unknown", "observation": _interop_catalogue_failure(),
             "stall_class": "native_catalogue_read_unavailable", "incident_since": 0,
             "next_action_at": 0, "infrastructure_budget": {"status": "available"}}
    assert fleet.select_action(state, board, 100) == ""


def test_infrastructure_tick_enqueues_one_bounded_repair_and_keeps_its_circuit(isolated, tmp_path, monkeypatch):
    _, fleet = isolated
    board = _interop_infrastructure_board(tmp_path)
    root = tmp_path / "watch"
    calls = []
    def runner(spec, **kwargs):
        calls.append(spec["argv"])
        if spec == board["probe"]:
            return {"returncode": 0, "stdout": json.dumps(_interop_catalogue_failure())}
        assert spec["argv"][:1] == ["private-repair"]
        return {"returncode": 0, "stdout": "{}"}
    monkeypatch.setattr(fleet, "command", lambda *a, **k: pytest.fail("no native service effect"))
    first = fleet.tick_board(board, root, apply=True, runner=runner, now=100)
    assert first["last_action"] == "repair" and first["stall_class"] == "native_catalogue_read_unavailable"
    assert first["infrastructure_budget"]["actions"] == 1
    budget = root / "private/infrastructure-budget.json"
    before = budget.read_bytes()
    second = fleet.tick_board(board, root, apply=True, now=1000,
        runner=lambda *a, **k: {"returncode": 0, "stdout": json.dumps(_interop_catalogue_failure(1000))})
    assert second["planned_action"] == "" and second["infrastructure_budget"]["status"] == "open"
    assert second["infrastructure_budget"]["total_actions"] == 1 and budget.read_bytes() == before
    assert len(calls) == 2 and calls[0] == ["private-probe"]


@pytest.mark.parametrize("mode", ["unknown_callback", "owner_policy", "invalid_policy", "observe_only", "operator_hold"])
def test_infrastructure_tick_preserves_denied_custody_before_cancellation_or_repair(isolated, tmp_path, monkeypatch, mode):
    _, fleet = isolated
    board = _interop_infrastructure_board(tmp_path)
    board["recovery_policy"]["stop_own_repair_on_wait"] = True
    sample = _interop_catalogue_failure()
    if mode == "unknown_callback": sample["details"]["selection_idle_reason"] = "expired_attempt_settlement_unavailable"
    elif mode == "owner_policy": board["recovery_policy"] = policy()
    elif mode == "invalid_policy": board["recovery_policy"] = {"mode": "restart"}
    elif mode == "observe_only": board["observation_only"] = True
    else:
        hold = tmp_path / "OPERATOR_STOP";hold.write_text("retained")
        board["hold_files"] = [str(hold)]
    root = tmp_path / "watch"
    prior = {"health": "unknown", "attempts": 3, "ensure_attempts": 2,
             "next_action_at": 9999, "pending_action": "repair", "last_action_at": 50}
    fleet.write_json(root / "private/state.json", prior)
    job = root / "repairs/private/job.json"
    fleet.write_json(job, {"status": "running", "attempts": 3, "next_attempt_at": 9999})
    before = job.read_bytes();calls = []
    def probe_only(spec, **kwargs):
        calls.append(spec["argv"]);assert spec == board["probe"]
        return {"returncode": 0, "stdout": json.dumps(sample)}
    monkeypatch.setattr(fleet, "command", lambda *a, **k: pytest.fail("wait cannot stop a repair cgroup"))
    result = fleet.tick_board(board, root, apply=True, runner=probe_only, now=100)
    assert result["planned_action"] == "" and calls == [["private-probe"]]
    assert job.read_bytes() == before
    assert all(result[k] == prior[k] for k in ("attempts", "ensure_attempts", "next_action_at", "pending_action"))
    assert not (root / "private/infrastructure-budget.json").exists()
    if mode == "unknown_callback": assert result["stall_class"] == "retained_attempt_settlement_wait"


def test_mixed_fleet_config_admits_both_closed_policies_without_changing_them(isolated, tmp_path):
    _, fleet = isolated
    a = _interop_infrastructure_board(tmp_path)
    b = {"id": "retained", "cwd": str(tmp_path), "probe": {"argv": ["private-probe"]},
         "recovery_policy": policy()}
    config = {"state_dir": str(tmp_path / "watch"), "boards": [a, b]}
    p = tmp_path / "fleet.json";p.write_text(json.dumps(config))
    assert fleet.load_config(p) == config
