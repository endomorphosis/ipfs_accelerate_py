from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.project_adapters import worldwide_legal_search_host_lifecycle as host
from ipfs_accelerate_py.agent_supervisor.project_adapters.worldwide_legal_search_watchdog import WorldwideWatchdogBridge


@pytest.fixture
def registered(tmp_path, monkeypatch):
    repository, state, home = tmp_path / "repo", tmp_path / "state", tmp_path / "home"
    repository.mkdir()
    state.mkdir()
    home.mkdir()
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
    config_path = repository / "config/worldwide_supervisor_host.json"
    config_path.parent.mkdir()
    config = {"schema": "worldwide-legal-search.host@1", "site_root": str(repository), "state_root": str(state),
              "runtime_argv": ["python", "-m", host.MODULE, "run", "--config", str(config_path)]}
    config_path.write_text(json.dumps(config))
    fragment = home / ".config/systemd/user" / host.UNIT
    fragment.parent.mkdir(parents=True)
    fragment.write_text("[Service]\nWorkingDirectory=" + str(repository) + "\nExecStart=python -m " + host.MODULE + " run --config " + str(config_path) + "\nKillMode=control-group\n")
    clock = [10000.0]
    native = {"pid": 101, "ready": False, "wrong_identity": False}
    group = "/user.slice/user-1000.slice/user@1000.service/app.slice/" + host.UNIT
    calls = []

    def process(pid):
        return {"pid": pid, "alive": pid > 0 and pid == native["pid"], "start_ticks": pid * 10, "uid": os.getuid(),
                "argv": ["python", "-m", "other.module" if native["wrong_identity"] else host.MODULE, "run", "--config", str(config_path)], "cgroups": [group]}

    monkeypatch.setattr(host, "process_identity", process)

    def command(argv, timeout):
        calls.append(argv)
        assert argv[:2] == ["systemctl", "--user"]
        assert host.UNIT in argv
        if argv[2] == "show":
            return "\n".join(["Id=" + host.UNIT, "MainPID=" + str(native["pid"]), "ControlGroup=" + (group if native["pid"] else ""), "FragmentPath=" + str(fragment), "ActiveState=" + ("active" if native["pid"] else "inactive"), "SubState=" + ("running" if native["pid"] else "dead"), "KillMode=control-group"])
        if argv[2] == "stop":
            native["pid"] = 0
        elif argv[2] == "start":
            native.update(pid=202, ready=True)
            projection_path = state / "campaign-status.json"
            projection = json.loads(projection_path.read_text()) if projection_path.exists() else {}
            projection.update(run_id="wwl-test", owner_pid=202, observed_at=clock[0])
            projection_path.write_text(json.dumps(projection))
        else:
            pytest.fail("unexpected command")
        return ""

    lifecycle = host.RegisteredSystemdLifecycle(repository=repository, state_root=state, run_id="wwl-test", command=command,
                clock=lambda: clock[0], monotonic=lambda: clock[0], sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    monkeypatch.setattr(lifecycle, "_cgroup_empty", lambda value: True)
    return lifecycle, native, calls, config_path, clock


def test_registered_host_recovery_fences_then_starts_only_exact_unit(registered):
    lifecycle, native, calls, config_path, clock = registered
    action = {"action_id": "a" * 64, "run_id": "wwl-test", "policy_id": "wwl-policy", "kind": "recover"}
    result = lifecycle.recover(action)
    assert result["recovered"] is True
    assert result["old_owner_fenced"] is True
    assert result["receipt"]
    mutations = [item for item in calls if item[2] != "show"]
    assert mutations == [["systemctl", "--user", "stop", host.UNIT], ["systemctl", "--user", "start", host.UNIT]]
    assert native["pid"] == 202
    # A repeated action consumes the native durable receipt, not another stop.
    calls.clear()
    repeated = lifecycle.recover(action)
    assert repeated["deduplicated"] is True
    assert not [item for item in calls if item[2] != "show"]


def test_lost_owner_can_be_recovered_without_owner_rpc(registered):
    lifecycle, native, calls, config_path, clock = registered
    native["pid"] = 0
    result = lifecycle.recover({"action_id": "b" * 64, "run_id": "wwl-test", "policy_id": "wwl-policy", "kind": "recover"})
    assert result["recovered"] is True
    assert native["pid"] == 202


def test_wrong_runtime_owner_is_never_stopped(registered):
    lifecycle, native, calls, config_path, clock = registered
    native["wrong_identity"] = True
    with pytest.raises(host.HostLifecycleError, match="MainPID"):
        lifecycle.recover({"action_id": "c" * 64, "run_id": "wwl-test", "policy_id": "wwl-policy", "kind": "recover"})
    assert not [item for item in calls if item[2] != "show"]


def test_other_service_registration_is_rejected_before_control(registered):
    lifecycle, native, calls, config_path, clock = registered
    config = json.loads(config_path.read_text())
    config["supervisor_unit"] = "another-important-service.service"
    config_path.write_text(json.dumps(config))
    calls.clear()
    with pytest.raises(host.HostLifecycleError, match="exact registered"):
        host.RegisteredSystemdLifecycle(repository=lifecycle.repository, state_root=lifecycle.state_root, run_id="wwl-test")
    assert calls == []


def test_other_campaign_action_is_rejected_before_control(registered):
    lifecycle, native, calls, config_path, clock = registered
    with pytest.raises(host.HostLifecycleError, match="registered campaign"):
        lifecycle.recover({"action_id": "d" * 64, "run_id": "different-campaign", "policy_id": "wwl-policy", "kind": "recover"})
    assert not [item for item in calls if item[2] != "show"]


def test_adapter_dispatch_recovers_a_dead_owner_and_preserves_single_authority(registered):
    lifecycle, native, calls, config_path, clock = registered
    state = lifecycle.state_root
    native["pid"] = 0
    (state / "campaign-readiness.json").write_text(json.dumps({"schema": "wwl.watchdog.ready@1", "run_id": "wwl-test", "repository": str(lifecycle.repository), "policy_id": "wwl-policy", "native_revision": "accepted", "admitted_plan_revision": "plan-1", "required_task_ids": ["WWL-000"], "bindings": {"reconcile": True, "lifecycle_restart": False, "repair_dispatch": False}}))
    (state / "campaign-status.json").write_text(json.dumps({"schema": "wwl.watchdog.snapshot@1", "run_id": "wwl-test", "owner_pid": 101, "observed_at": 9700, "heartbeat_at": 9700,
        "progress_token": "last-real-phase", "authority_revision": "source-1", "accepted_tree": "tree-1", "hold": False}))
    bridge = WorldwideWatchdogBridge(repository=lifecycle.repository, state_root=state, run_id="wwl-test", clock=lambda: clock[0], host_lifecycle=lifecycle)
    request = {"schema": "wwl.watchdog.action@1", "action_id": "e" * 64, "run_id": "wwl-test", "policy_id": "wwl-policy", "kind": "recover",
        "issued_at": clock[0], "authority_revision": "source-1", "accepted_tree": "tree-1"}
    result = bridge.dispatch(request, timeout_seconds=0)
    assert result["status"] == "accepted"
    assert not (state / "control/requests").exists()
    observed = bridge.observe({"run_id": "wwl-test", "policy_id": "wwl-policy", "action_ids": [request["action_id"]]})
    assert observed["action_observations"][0]["status"] == "succeeded"
    assert observed["action_observations"][0]["effects_verified"] is True
    calls.clear()
    assert bridge.dispatch(request, timeout_seconds=0) == result
    assert not [item for item in calls if item[2] != "show"]


def test_stale_task_projection_keeps_old_heartbeat_but_gets_real_host_observation(registered):
    lifecycle, native, calls, config_path, clock = registered
    state = lifecycle.state_root
    (state / "campaign-readiness.json").write_text(json.dumps({"schema": "wwl.watchdog.ready@1", "run_id": "wwl-test", "repository": str(lifecycle.repository), "policy_id": "wwl-policy", "native_revision": "accepted", "admitted_plan_revision": "plan-1", "required_task_ids": ["WWL-000"], "bindings": {"lifecycle_restart": True}}))
    snapshot = {"schema": "wwl.watchdog.snapshot@1", "run_id": "wwl-test", "owner_pid": 101, "observed_at": 9700, "heartbeat_at": 9700,
                "progress_token": "last-real-phase", "authority_revision": "source-1", "completion_verified": True, "completion_receipt": "old-root-proof"}
    (state / "campaign-status.json").write_text(json.dumps(snapshot))
    native["pid"] = 0
    bridge = WorldwideWatchdogBridge(repository=lifecycle.repository, state_root=state, run_id="wwl-test", clock=lambda: clock[0], host_lifecycle=lifecycle)
    observed = bridge.observe({"run_id": "wwl-test", "policy_id": "wwl-policy"})
    assert observed["observed_at"] == 10000
    assert observed["projection_observed_at"] == 9700
    assert observed["heartbeat_at"] == 9700
    assert observed["progress_token"] == "last-real-phase"
    assert observed["owner_alive"] is False
    assert observed["source_projection_stale"] is True
    assert observed["completion_verified"] is False
    assert not observed["completion_receipt"]
