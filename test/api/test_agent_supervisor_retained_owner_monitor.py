"""Hermetic deny-only owner retention checks for the standalone service monitor."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


POLICY = {"schema": "agent-supervisor/owner-retention@1", "mode": "retain",
          "profile_id": "retained-native-owner", "reason": "Preserve native attempt custody"}
NOW = 1000.0


@pytest.fixture
def monitor_module(monkeypatch: pytest.MonkeyPatch):
    path = Path(__file__).resolve().parents[2] / "ipfs_accelerate_py/agent_supervisor/rescue/board_service_monitor.py"
    spec = importlib.util.spec_from_file_location("_retained_owner_monitor_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    def forbidden(*args, **kwargs):
        pytest.fail("unexpected host process or systemd effect")
    monkeypatch.setattr(module.subprocess, "run", forbidden)
    monkeypatch.setattr(module, "systemd_show", forbidden)
    monkeypatch.setattr(module, "_process_snapshot", forbidden)
    monkeypatch.setattr(module.time, "time", lambda: NOW)
    monkeypatch.setattr(module.time, "monotonic", lambda: NOW)
    return module


@pytest.fixture
def config(tmp_path: Path, monitor_module):
    path = tmp_path / "monitor.json"
    path.write_text(json.dumps({"service": "fixture-sawm.service", "state_dir": str(tmp_path),
        "state_prefix": "fixture", "board_namespace": "fixture-native", "startup_grace_seconds": 10,
        "heartbeat_timeout_seconds": 30, "progress_timeout_seconds": 100}), encoding="utf-8")
    return monitor_module.load_config(path)


def _case(name: str):
    service = {"LoadState": "loaded", "ActiveState": "active", "MainPID": "123",
               "ExecMainStartTimestampMonotonic": "1000000"}
    evidence = {"implement": True, "state_available": True, "active": False,
                "ready_count": 0, "last_heartbeat_at_epoch": 0, "last_progress_at_epoch": 0}
    previous = {"first_seen_at_epoch": 1.0}
    expected = name
    if name == "service_dead":
        service.update(ActiveState="inactive", MainPID="0", ExecMainStartTimestampMonotonic="0")
    elif name == "stale_worker":
        evidence.update(active=True, active_started_at_epoch=950.0)
    elif name == "worker_deadline_exceeded":
        evidence.update(active=True, active_started_at_epoch=100.0, worker_heartbeat_at_epoch=NOW)
    elif name == "ready_backlog_stalled":
        evidence.update(ready_count=1, last_heartbeat_at_epoch=NOW)
        previous["ready_since_at_epoch"] = 100.0
    elif name == "native_recovery_stalled":
        evidence["native_recovery"] = {"expired": True, "quiescent": True}
    return service, evidence, previous, expected


def _stub_observations(monkeypatch, module, service, observations):
    records = iter(observations)
    monkeypatch.setattr(module, "collect_evidence", lambda *_: dict(next(records)))
    monkeypatch.setattr(module, "systemd_show", lambda *_: dict(service))
    monkeypatch.setattr(module, "_native_recovery_observation", lambda c, s, e, p, n, t: e.get("native_recovery", {}))


@pytest.mark.parametrize("condition", ["service_dead", "stale_heartbeat", "stale_worker",
    "worker_deadline_exceeded", "ready_backlog_stalled", "native_recovery_stalled"])
def test_retained_owner_monitor_denies_every_ordinary_restart(config, monitor_module, condition):
    module = monitor_module
    service, evidence, previous, expected = _case(condition)
    ordinary = module.evaluate_health(config, service, evidence, previous, NOW, NOW)
    assert ordinary["restart_eligible"] and ordinary["status"] == expected
    config["recovery_policy"] = dict(POLICY)
    retained = module.evaluate_health(config, service, evidence, previous, NOW, NOW)
    assert retained["status"] == "native_owner_retention_required"
    assert retained["restart_eligible"] is False
    assert retained["potential_repair_recommended"] is True
    assert retained["repair_authorized"] is False
    assert retained["completion_authority"] is False


@pytest.mark.parametrize("policy", [None, {}, {**POLICY, "mode": "restart"},
    {**POLICY, "profile_id": ""}, {**POLICY, "extra": True}])
def test_retained_owner_monitor_invalid_policy_fails_closed(config, monitor_module, policy):
    config["recovery_policy"] = policy
    service, evidence, previous, _ = _case("service_dead")
    result = monitor_module.evaluate_health(config, service, evidence, previous, NOW, NOW)
    assert result["status"] == "invalid_owner_retention_policy"
    assert result["restart_eligible"] is False


def test_retained_owner_monitor_settlement_wait_without_config_policy(config, monitor_module):
    service, evidence, previous, _ = _case("stale_heartbeat")
    evidence["selection_idle_reason"] = "expired_attempt_settlement_unavailable"
    result = monitor_module.evaluate_health(config, service, evidence, previous, NOW, NOW)
    assert result["status"] == "retained_attempt_settlement_wait"
    assert result["restart_eligible"] is False
    assert result["repair_authorized"] is False


def test_retained_owner_monitor_preserves_top_level_service_policy(tmp_path, monitor_module):
    path = tmp_path / "configured.json"
    path.write_text(json.dumps({"service": "one.service", "state_dir": str(tmp_path),
        "state_prefix": "one", "board_namespace": "one", "recovery_policy": POLICY}), encoding="utf-8")
    first = monitor_module.load_config(path)
    assert first["recovery_policy"] == POLICY
    assert monitor_module.owner_retention_required(first)
    path.write_text(json.dumps({"service": "two.service", "state_dir": str(tmp_path),
        "state_prefix": "two", "board_namespace": "two"}), encoding="utf-8")
    assert not monitor_module.owner_retention_required(monitor_module.load_config(path))


@pytest.mark.parametrize("location", ["native", "status", "heartbeat"])
def test_retained_owner_monitor_collects_settlement_marker_from_each_metadata_source(
    config, monitor_module, location
):
    native = {"board_namespace": config["board_namespace"], "updated_at": NOW}
    records = {"native": native, "status": {"updated_at": NOW}, "heartbeat": {"completed_at": NOW}}
    records[location]["selection_idle_reason"] = "expired_attempt_settlement_unavailable"
    for source, key in (("native", "native_state_path"), ("status", "status_path"), ("heartbeat", "heartbeat_path")):
        Path(config[key]).write_text(json.dumps(records[source]), encoding="utf-8")
    evidence = monitor_module.collect_evidence(config, NOW)
    assert evidence["selection_idle_reason"] == "expired_attempt_settlement_unavailable"
    service, _, previous, _ = _case("stale_heartbeat")
    assert monitor_module.evaluate_health(config, service, evidence, previous, NOW, NOW)["restart_eligible"] is False


def test_retained_owner_monitor_never_charges_initial_retention(config, monitor_module, monkeypatch):
    module = monitor_module
    config["recovery_policy"] = dict(POLICY)
    service, evidence, _, _ = _case("stale_heartbeat")
    _stub_observations(monkeypatch, module, service, [evidence])
    result = module.monitor(config, apply=True)
    assert result["status"] == "native_owner_retention_required"
    assert result["action"] == "none"
    assert result["restart_attempts_last_hour"] == 0
    assert json.loads(Path(config["monitor_state_path"]).read_text())["restart_attempts_at_epoch"] == []


def test_retained_owner_monitor_refresh_retention_does_not_charge(config, monitor_module, monkeypatch):
    module = monitor_module
    service, evidence, _, _ = _case("stale_heartbeat")
    refreshed = {**evidence, "selection_idle_reason": "expired_attempt_settlement_unavailable"}
    _stub_observations(monkeypatch, module, service, [evidence, refreshed])
    result = module.monitor(config, apply=True)
    assert result["status"] == "retained_attempt_settlement_wait"
    assert result["action"] == "restart_suppressed_health_changed"
    assert result["restart_attempts_last_hour"] == 0


def test_retained_owner_monitor_final_effect_guard_is_independent(config, monitor_module, monkeypatch):
    module = monitor_module
    config["recovery_policy"] = dict(POLICY)
    service, evidence, _, _ = _case("stale_heartbeat")
    _stub_observations(monkeypatch, module, service, [evidence, evidence])
    # Simulate an obsolete evaluator; the effect boundary must still refuse.
    monkeypatch.setattr(module, "evaluate_health", lambda *_: {
        "status": "stale_heartbeat", "reason": "old diagnosis", "healthy": False, "restart_eligible": True})
    result = module.monitor(config, apply=True)
    assert result["action"] == "restart_suppressed_retention"
    assert result["status"] == "native_owner_retention_required"
    assert result["restart_attempts_last_hour"] == 0
    assert json.loads(Path(config["monitor_state_path"]).read_text())["restart_attempts_at_epoch"] == []


def test_retained_owner_monitor_ordinary_restart_keeps_charged_budget(config, monitor_module, monkeypatch):
    module = monitor_module
    service, evidence, _, _ = _case("stale_heartbeat")
    _stub_observations(monkeypatch, module, service, [evidence, evidence])
    calls = []
    def restart(argv, **kwargs):
        assert json.loads(Path(config["monitor_state_path"]).read_text())["restart_attempts_at_epoch"] == [NOW]
        calls.append(argv)
        return SimpleNamespace(returncode=0, stderr="")
    monkeypatch.setattr(module.subprocess, "run", restart)
    result = module.monitor(config, apply=True)
    assert calls == [["systemctl", "--user", "restart", config["service"]]]
    assert result["action"] == "restart_requested"
    assert result["restart_attempts_last_hour"] == 1
