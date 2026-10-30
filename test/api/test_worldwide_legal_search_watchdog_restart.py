"""WWL-105: loop and supervisor restart during in-flight recovery.

Deduplication, cooldown, attempt accounting and stop/hold markers survive.
The optional systemd guardian may revive only the registered watchdog loop.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.project_adapters import worldwide_legal_search_watchdog as watchdog
from ipfs_accelerate_py.agent_supervisor.project_adapters.worldwide_legal_search_watchdog import (
    CANONICAL_PYTHON, GUARDIAN_RESTART_BOUND, GUARDIAN_WINDOW_SECONDS, LOOP_UNIT, SUPERVISOR_MODULE,
    SUPERVISOR_UNIT, WatchdogBridgeError, WorldwideWatchdogBridge, main, read_owned_json,
)

RUN = "wwl-test"
POLICY = "wwl-policy"
NOW = 10000
COOLDOWN = 10300
ACTION_ID = "a" * 64
OTHER_ID = "b" * 64


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def action(action_id: str = ACTION_ID) -> dict:
    return {
        "schema": "wwl.watchdog.action@1", "action_id": action_id, "run_id": RUN, "policy_id": POLICY,
        "kind": "recover", "authority_revision": "task-source-1", "accepted_tree": "accepted-tree-1",
        "issued_at": NOW, "incident_id": "incident-1", "reason": "owner_absent", "task_id": "WWL-000",
        "stage": "implementation",
    }


def loop_state(body: dict, *, recover: int = 1, reconcile: int = 2, cooldown: int = COOLDOWN) -> dict:
    return {
        "schema": "wwl.watchdog.state@1", "run_id": RUN, "policy_digest": "digest-1",
        "actions": 4, "repairs": 1, "required_task_ids": ["WWL-000", "WWL-004"],
        "incidents": {"incident-1": {"reconcile": reconcile, "recover": recover, "diagnose": 0, "repair": 1, "next_action_at": cooldown}},
        "outbox": body, "inflight": {}, "uncertain_actions": {},
    }


class Host:
    def __init__(self):
        self.recover_calls = 0

    def probe(self) -> dict:
        return {
            "unit": SUPERVISOR_UNIT, "registered": True, "main_pid": 0, "start_ticks": 0,
            "owner_alive": False, "owner_identity_verified": True, "control_group": "",
            "active_state": "inactive", "fresh_owner_projection": False, "observed_at": NOW,
        }

    def recover(self, request: dict) -> dict:
        self.recover_calls += 1
        return {"recovered": False, "receipt": "native:inflight-recovery", "old_owner_fenced": False, "deduplicated": False}


def campaign(tmp_path: Path):
    repository, state = tmp_path / "repo", tmp_path / "state"
    repository.mkdir()
    state.mkdir()
    write(state / "campaign-readiness.json", {
        "schema": "wwl.watchdog.ready@1", "run_id": RUN, "repository": str(repository),
        "policy_id": POLICY, "native_revision": "accepted-native-revision", "admitted_plan_revision": "plan-1",
        "required_task_ids": ["WWL-000", "WWL-004"], "readiness_receipt": "native:actual-test-evidence",
        "bindings": {"task_source": True, "meaningful_progress": True, "reconcile": True, "lifecycle_restart": True, "repair_dispatch": False, "incident_deduplication": True},
    })
    write(state / "campaign-status.json", {
        "schema": "wwl.watchdog.snapshot@1", "run_id": RUN, "observed_at": NOW,
        "authority_revision": "task-source-1", "accepted_tree": "accepted-tree-1", "hold": False,
        "admitted_plan_revision": "plan-1", "heartbeat_at": NOW, "progress_token": "validated-phase-1",
        "required_task_ids": ["WWL-000", "WWL-004"],
    })
    return repository, state


def open_bridge(repository: Path, state: Path, host: Host) -> WorldwideWatchdogBridge:
    return WorldwideWatchdogBridge(repository=repository, state_root=state, run_id=RUN, clock=lambda: NOW, host_lifecycle=host)


def publish_inflight(repository: Path, state: Path, host: Host, body: dict) -> WorldwideWatchdogBridge:
    write(state / "watchdog" / "state.json", loop_state(body))
    bridge = open_bridge(repository, state, host)
    result = bridge.dispatch(body, timeout_seconds=0)
    assert result["status"] == "unknown"
    assert result["receipt"] == "native:inflight-recovery"
    assert host.recover_calls == 1
    return bridge


def test_restart_preserves_inflight_dedup_cooldown_and_attempts(tmp_path: Path):
    repository, state = campaign(tmp_path)
    host = Host()
    body = action()
    publish_inflight(repository, state, host, body)
    before = (state / "watchdog" / "state.json").read_bytes()
    journal_before = (state / "watchdog" / "restart-journal.json").read_bytes()

    restarted = open_bridge(repository, state, host)
    resumed = restarted.resume_after_restart()
    assert resumed["schema"] == "wwl.watchdog.restart-resume@1"
    assert resumed["loop_restarted"] is True and resumed["supervisor_restarted"] is True
    assert resumed["deduplicated"] is True
    assert resumed["attempts_refunded"] is False and resumed["cooldown_reset"] is False
    assert resumed["hold_cleared"] is False and resumed["completion_authority"] is False
    assert resumed["inflight"] == [{
        "action_id": ACTION_ID, "incident_id": "incident-1", "kind": "recover",
        "attempt_accounting": {"reconcile": 2, "recover": 1, "diagnose": 0, "repair": 1, "actions": 4, "repairs": 1},
        "cooldown_until": float(COOLDOWN),
    }]
    assert (state / "watchdog" / "state.json").read_bytes() == before
    assert (state / "watchdog" / "restart-journal.json").read_bytes() == journal_before

    replay = restarted.dispatch(body, timeout_seconds=0)
    assert replay["status"] == "unknown" and replay["receipt"] == "native:inflight-recovery"
    assert host.recover_calls == 1
    refused = restarted.dispatch(action(OTHER_ID), timeout_seconds=0)
    assert refused["status"] == "blocked"
    refusal = read_owned_json(state / "control" / "refusals" / (OTHER_ID + ".json"))
    assert refusal["reason"] == "durable_incident_already_claimed"
    assert refusal["effects_applied"] is False
    assert host.recover_calls == 1
    assert (state / "watchdog" / "state.json").read_bytes() == before


def test_stop_hold_survives_supervisor_restart_and_a_missing_marker_does_not(tmp_path: Path):
    repository, state = campaign(tmp_path)
    host = Host()
    body = action()
    publish_inflight(repository, state, host, body)
    (state / "STOP").write_text("operator stop\n", encoding="utf-8")
    restarted = open_bridge(repository, state, host)
    observed = restarted.observe({"policy_id": POLICY, "run_id": RUN})
    assert observed["hold"] is True
    assert read_owned_json(state / "watchdog" / "restart-journal.json")["sealed_hold_markers"] == ["STOP"]
    resumed = restarted.resume_after_restart()
    assert resumed["hold_markers"] == ["STOP"]
    assert resumed["hold_cleared"] is False
    blocked = restarted.dispatch(action(OTHER_ID), timeout_seconds=0)
    assert blocked["status"] == "blocked"
    assert (state / "STOP").read_text(encoding="utf-8") == "operator stop\n"
    assert host.recover_calls == 1

    (state / "STOP").unlink()
    with pytest.raises(WatchdogBridgeError, match="stop/hold did not survive restart"):
        restarted.resume_after_restart()
    assert host.recover_calls == 1


def test_hold_added_after_the_seal_still_blocks_the_restarted_supervisor(tmp_path: Path):
    repository, state = campaign(tmp_path)
    host = Host()
    publish_inflight(repository, state, host, action())
    (state / "OPERATOR_STOP").write_text("later hold\n", encoding="utf-8")
    restarted = open_bridge(repository, state, host)
    assert restarted.resume_after_restart()["hold_markers"] == ["OPERATOR_STOP"]
    assert restarted.dispatch(action(OTHER_ID), timeout_seconds=0)["status"] == "blocked"
    assert (state / "OPERATOR_STOP").read_text(encoding="utf-8") == "later hold\n"


@pytest.mark.parametrize(("mutate", "message"), [
    (lambda state: state["incidents"]["incident-1"].__setitem__("recover", 0), "attempt accounting regressed"),
    (lambda state: state["incidents"]["incident-1"].__setitem__("next_action_at", NOW), "cooldown moved earlier"),
    (lambda state: state.__setitem__("outbox", None), "lost the in-flight recovery"),
    (lambda state: state.__setitem__("required_task_ids", ["WWL-000"]), "pruned admitted tasks"),
])
def test_restart_rejects_a_refunded_loop_ledger(tmp_path: Path, mutate, message: str):
    repository, state = campaign(tmp_path)
    host = Host()
    publish_inflight(repository, state, host, action())
    sealed = read_owned_json(state / "watchdog" / "restart-journal.json")
    original = (state / "watchdog" / "state.json").read_text(encoding="utf-8")
    tampered = json.loads(original)
    mutate(tampered)
    write(state / "watchdog" / "state.json", tampered)
    restarted = open_bridge(repository, state, host)
    with pytest.raises(WatchdogBridgeError, match=message):
        restarted.resume_after_restart()
    assert read_owned_json(state / "watchdog" / "restart-journal.json") == sealed
    assert host.recover_calls == 1


def test_same_action_cannot_lower_the_sealed_cooldown_on_replay(tmp_path: Path):
    repository, state = campaign(tmp_path)
    host = Host()
    body = action()
    bridge = publish_inflight(repository, state, host, body)
    shortened = loop_state(body, cooldown=NOW)
    write(state / "watchdog" / "state.json", shortened)
    with pytest.raises(WatchdogBridgeError, match="cooldown moved earlier"):
        bridge._seal_published_action(body, status="inflight")
    assert read_owned_json(state / "watchdog" / "restart-journal.json")["actions"][ACTION_ID]["cooldown_until"] == float(COOLDOWN)


def guardian_for(tmp_path: Path, state: Path, repository: Path, *, clock=lambda: NOW, show=None, command=None):
    loop_script = tmp_path / "implementation_plan" / "worldwide_legal_search" / "watchdog_loop.py"
    loop_script.parent.mkdir(parents=True, exist_ok=True)
    loop_script.write_text("# registered loop\n", encoding="utf-8")
    fragment = tmp_path / "unit" / watchdog.LOOP_UNIT
    guardian = watchdog.OptionalSystemdLoopGuardian(
        repository=repository, state_root=state, run_id=RUN, loop_script=loop_script,
        clock=clock, show=show, command=command, fragment_path=fragment,
    )
    fragment.parent.mkdir(parents=True, exist_ok=True)
    fragment.write_text(guardian.render_unit(), encoding="utf-8")
    return guardian, fragment


def shown(fragment: Path, *, unit: str = LOOP_UNIT, active: str = "inactive", pid: str = "0") -> dict[str, str]:
    return {"Id": unit, "FragmentPath": str(fragment), "ActiveState": active, "MainPID": pid}


def test_guardian_revives_only_the_registered_loop(tmp_path: Path):
    repository, state = campaign(tmp_path)
    calls: list[list[str]] = []

    def command(argv, timeout):
        calls.append(list(argv))
        assert timeout == 5
        return ""

    guardian, fragment = guardian_for(tmp_path, state, repository, show=lambda: shown(fragment), command=command)
    text = guardian.render_unit()
    assert text.count("\nExecStart=") == 1
    assert "watchdog_loop.py" in text and "guardian-start-check" in text
    assert "Restart=on-failure" in text and "StartLimitBurst=" + str(GUARDIAN_RESTART_BOUND) in text
    assert "StartLimitIntervalSec=" + str(int(GUARDIAN_WINDOW_SECONDS)) in text
    assert CANONICAL_PYTHON in text
    assert SUPERVISOR_MODULE not in text and SUPERVISOR_UNIT not in text
    assert str(state / "watchdog" / "state.json") in text

    check = guardian.start_check()
    assert check["may_start"] is True and check["reason"] == "registered_loop_start_allowed"
    assert check["started_units"] == [] and calls == []
    revived = guardian.revive()
    assert revived["revived"] is True and revived["started_units"] == [LOOP_UNIT]
    assert calls == [["systemctl", "--user", "start", LOOP_UNIT]]
    journal = read_owned_json(state / "watchdog" / "restart-journal.json")
    assert journal["guardian_restarts"] == [float(NOW)]
    assert journal["completion_authority"] is False


def test_guardian_refuses_another_unit_a_hold_and_a_live_loop(tmp_path: Path):
    repository, state = campaign(tmp_path)
    calls: list[list[str]] = []
    guardian, fragment = guardian_for(
        tmp_path, state, repository,
        show=lambda: shown(fragment, unit=SUPERVISOR_UNIT),
        command=lambda argv, timeout: calls.append(list(argv)) or "",
    )
    foreign = guardian.start_check()
    assert foreign["may_start"] is False and foreign["reason"] == "unregistered_unit"
    assert guardian.revive()["revived"] is False and calls == []

    fragment.write_text(guardian.render_unit().replace("watchdog_loop.py", SUPERVISOR_MODULE), encoding="utf-8")
    guardian.show = lambda: shown(fragment)
    mismatched = guardian.start_check()
    assert mismatched["may_start"] is False and mismatched["reason"] == "fragment_does_not_match_registered_loop"
    assert calls == []

    fragment.write_text(guardian.render_unit(), encoding="utf-8")
    (state / "HOLD").write_text("pause\n", encoding="utf-8")
    held = guardian.revive()
    assert held["may_start"] is False and held["reason"] == "hold_blocks_start"
    assert held["hold_markers"] == ["HOLD"] and held["revived"] is False and calls == []
    assert not (state / "watchdog" / "restart-journal.json").exists()
    (state / "HOLD").unlink()

    guardian.show = lambda: shown(fragment, active="active", pid="4242")
    running = guardian.revive()
    assert running["reason"] == "registered_loop_already_running" and running["revived"] is False
    assert calls == []


def test_guardian_restart_bound_ignores_stale_starts_and_keeps_a_failed_charge(tmp_path: Path):
    repository, state = campaign(tmp_path)
    calls: list[list[str]] = []
    guardian, fragment = guardian_for(tmp_path, state, repository, show=lambda: shown(fragment), command=lambda argv, timeout: calls.append(argv) or "")
    stale = watchdog.empty_restart_journal(RUN)
    stale["guardian_restarts"] = [float(NOW - GUARDIAN_WINDOW_SECONDS - 5), float(NOW - 10), float(NOW - 20)]
    write(state / "watchdog" / "restart-journal.json", stale)
    assert guardian.start_check()["may_start"] is True
    assert guardian.start_check()["restarts_in_window"] == 2

    exhausted = watchdog.empty_restart_journal(RUN)
    exhausted["guardian_restarts"] = [float(NOW - 1), float(NOW - 2), float(NOW - 3)]
    write(state / "watchdog" / "restart-journal.json", exhausted)
    blocked = guardian.revive()
    assert blocked["reason"] == "restart_bound_exhausted" and blocked["revived"] is False
    assert calls == []
    assert read_owned_json(state / "watchdog" / "restart-journal.json")["guardian_restarts"] == exhausted["guardian_restarts"]

    write(state / "watchdog" / "restart-journal.json", watchdog.empty_restart_journal(RUN))

    def fail(argv, timeout):
        calls.append(list(argv))
        raise WatchdogBridgeError("systemctl failed")

    guardian.command = fail
    failed = guardian.revive()
    assert failed["revived"] is False and failed["reason"] == "registered_loop_start_failed"
    assert failed["restart_charged"] is True
    assert calls == [["systemctl", "--user", "start", LOOP_UNIT]]
    assert read_owned_json(state / "watchdog" / "restart-journal.json")["guardian_restarts"] == [float(NOW)]


def test_guardian_start_check_cli_is_hold_aware(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    repository, state = campaign(tmp_path)
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    loop_script = tmp_path / "watchdog_loop.py"
    loop_script.write_text("# loop\n", encoding="utf-8")
    guardian = watchdog.OptionalSystemdLoopGuardian(repository=repository, state_root=state, run_id=RUN, loop_script=loop_script, clock=lambda: NOW)
    fragment = home / ".config" / "systemd" / "user" / LOOP_UNIT
    fragment.parent.mkdir(parents=True)
    fragment.write_text(guardian.render_unit(), encoding="utf-8")
    calls: list[list[str]] = []

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        return SimpleNamespace(
            returncode=0,
            stdout="Id=%s\nFragmentPath=%s\nActiveState=inactive\nMainPID=0\n" % (LOOP_UNIT, fragment),
            stderr="",
        )

    monkeypatch.setattr(watchdog.subprocess, "run", fake_run)
    argv = ["--repository", str(repository), "--state-root", str(state), "--run-id", RUN, "--loop-script", str(loop_script), "guardian-start-check"]
    assert main(argv) == 0
    allowed = json.loads(capsys.readouterr().out)
    assert allowed["may_start"] is True and allowed["unit"] == LOOP_UNIT
    (state / "HOLD").write_text("pause\n", encoding="utf-8")
    assert main(argv) == 2
    refused = json.loads(capsys.readouterr().out)
    assert refused["may_start"] is False and refused["reason"] == "hold_blocks_start"
    assert calls and all(call[:3] == ["systemctl", "--user", "show"] for call in calls)
    assert all("start" not in call for call in calls)
