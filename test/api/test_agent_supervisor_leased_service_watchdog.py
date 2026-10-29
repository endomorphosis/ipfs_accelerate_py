"""A watchdog may recover only its exact leased service and bounded task work."""

from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
import os
import socket
from pathlib import Path
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.rescue.leased_service_watchdog import (
    Config, Host, LeasedServiceWatchdog, ObservationError, SCHEMA, parse_board,
)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class FakeHost(Host):
    def __init__(self, config, pointer_path, state_path, status_path):
        self.config, self.pointer_path, self.state_path, self.status_path = config, pointer_path, state_path, status_path
        self.now, self.restarts, self.dirt = 1000.0, 0, []
        self.processes = {}
        self.view = {"Id": config.unit, "ActiveState": "active", "SubState": "running", "MainPID": "100",
                     "FragmentPath": str(config.unit_fragment_path), "DropInPaths": "", "NeedDaemonReload": "no",
                     "WorkingDirectory": str(config.repo_root), "KillMode": "control-group", "ControlGroup": "/user/test.service",
                     "argv": list(config.expected_argv)}
        self.install_owner(100)

    def install_owner(self, base):
        c = self.config
        self.view["MainPID"] = str(base)
        self.processes = {}
        child_argv = list(c.expected_argv[c.expected_argv.index("--") + 1:])
        trampoline = ["/usr/bin/node", "-e", "/* registered lease trampoline fixture */"]

        def process(pid, ppid, argv, start_wall):
            value = {"pid": pid, "ppid": ppid, "argv": argv, "start_ticks": str(pid * 10), "uid": os.getuid(),
                     "boot_id": "boot", "pid_namespace": "pid:[fixture]", "cwd": str(c.repo_root),
                     "cgroups": [self.view["ControlGroup"]], "start_wall": start_wall, "cpu_ticks": 1,
                     "environment": {"IPFS_ACCELERATE_AGENT_MAX_DIRTY_ATTEMPTS": "0"}}
            self.processes[pid] = value
            return value

        main = process(base, 1, list(c.expected_argv), 800)
        protected = process(base + 1, base, trampoline, 801)
        process(base + 2, base + 1, child_argv, 802)
        process(base + 3, base + 2, ["python3", "produce.py", "--task-id", "TEST-001"], self.now - 100)
        process(base + 4, base + 3, ["grok", "--model", "grok-4.7"], self.now - 20)

        def identity(p):
            return {"method": "linux-procfs-starttime-v1", "startTicks": p["start_ticks"], "bootId": p["boot_id"],
                    "pidNamespace": p["pid_namespace"], "uid": p["uid"], "command": p["argv"]}

        pointer = {"schemaVersion": 1, "leaseId": f"lease-{base}",
                   "namespace": {"name": c.lease_namespace, "superproject": str(c.repo_root),
                                 "commonDirectory": str(c.lease_owner_path.parent.parent)},
                   "lane": {"id": c.lane_id, "board": "board.md"},
                   "owner": {"pid": base, "hostname": socket.gethostname(), "processIdentity": identity(main)},
                   "command": child_argv, "acquiredAt": "fixture"}
        write(self.pointer_path, pointer)
        write(self.state_path, {**pointer, "childPid": base + 1, "childProcessIdentity": identity(protected), "childProcessGroupId": base + 1})
        write(self.status_path, {"pid": base + 2, "active_task_id": "TEST-001", "phase": "deterministic_command",
                                 "heartbeat_at_ms": self.now * 1000, "last_task_transition_at_ms": 900000})

    def unit(self, _config):
        return copy.deepcopy(self.view)

    def process(self, pid, *, environment=False):
        return copy.deepcopy(self.processes.get(pid))

    def members(self, _cgroup):
        return [pid for pid, p in self.processes.items() if _cgroup in p["cgroups"]]

    def exact_processes(self, _config, _foreground):
        return [p["pid"] for p in self.processes.values() if "coordinator.py" in p["argv"]]

    def dirty_paths(self, _config):
        return list(self.dirt)

    def restart(self, _config):
        self.restarts += 1
        self.install_owner(100 + 100 * self.restarts)
        # Model a new service whose replacement task also reaches its bound.
        self.processes[103 + 100 * self.restarts]["start_wall"] = self.now - 200


@pytest.fixture
def case(tmp_path):
    root = tmp_path
    fragment = root / "service.unit"
    fragment.write_text("[Service]\nKillMode=control-group\n")
    argv = ("/usr/bin/node", "lease.js", "--run", "--lane", "test", "--board", "board.md", "--", "python3", "coordinator.py", "--implement")
    board = root / "board.md"
    board.write_text("## TEST-001 Work\n- Status: ready\n- Depends on:\n"
                     "- Deterministic command: python3 produce.py --task-id TEST-001\n"
                     "- Validation: python3 validate.py\n- Predicted files: output.json\n"
                     "- Implementation timeout seconds: 100\n")
    config = Config(unit="test.service", repo_root=root, lane_id="test", board_path=board,
                    task_prefix="## TEST-", status_path=root / "runtime/lane/status.json",
                    lease_owner_path=root / ".git/lease/owner.json", inventory_path=root / "inventory.json",
                    state_dir=root / "runtime/watchdog", unit_fragment_path=fragment,
                    unit_fragment_sha256=hashlib.sha256(fragment.read_bytes()).hexdigest(), expected_argv=argv,
                    log_path=root / "lane.log", startup_grace_seconds=10, deadline_grace_seconds=20,
                    idle_stall_seconds=30, restart_cooldown_seconds=25, max_task_timeout_seconds=2000)
    write(config.inventory_path, {"lanes": [{"id": "test", "board": "board.md", "taskPrefix": "## TEST-",
                                            "leaseRequiredForImplementation": True,
                                            "launch": {"commandPolicy": "exact", "command": ["node", *argv[1:]]}}]})
    host = FakeHost(config, config.lease_owner_path, config.lease_owner_path.parent / ".owner-state-fixture.json", config.status_path)
    watchdog = LeasedServiceWatchdog(config, host=host)

    def observe(now=1000, apply=False, restart_object=False):
        host.now = now
        monitor = LeasedServiceWatchdog(config, host=host) if restart_object else watchdog
        return monitor.observe(now=now, apply=apply)

    return SimpleNamespace(config=config, host=host, watchdog=watchdog, observe=observe, root=root)


def test_real_trampoline_shape_and_healthy_foreground_are_accepted(case):
    result = case.observe()
    assert result["action"] == "active_within_deadline"
    assert result["task_deadline_remaining_seconds"] == 20
    assert result["native_process_count"] == 1
    assert case.host.restarts == 0
    assert result["completion_authority"] is result["work_complete"] is False


def test_host_uses_supported_property_arguments_and_structured_execstart(case, monkeypatch):
    calls = []
    def run(argv, timeout=30):
        calls.append(argv)
        if argv[0] == "systemctl":
            assert "--property=Id" in argv and not any(arg.startswith("-p=") for arg in argv)
            return "Id=test.service\nMainPID=100\n"
        return json.dumps({"type": "a(sasbttttuii)", "data": [["/usr/bin/node", list(case.config.expected_argv), False]]})
    host = Host();monkeypatch.setattr(host, "_run", run)
    unit = host.unit(case.config)
    assert unit["Id"] == "test.service" and unit["argv"] == list(case.config.expected_argv)
    assert calls[1][0] == "busctl"


@pytest.mark.parametrize("tamper", ["main-birth", "protected-command", "foreground-command", "foreground-ancestry", "foreign-cgroup", "environment", "foreign-lease", "unit-argv", "fragment", "drop-in", "reload"])
def test_unverified_owners_and_configuration_never_authorize_restart(case, tamper):
    h, c = case.host, case.config
    if tamper == "main-birth": h.processes[100]["start_ticks"] = "reused"
    elif tamper == "protected-command": h.processes[101]["argv"] = ["unrelated"]
    elif tamper == "foreground-command": h.processes[102]["argv"] = ["python3", "other.py"]
    elif tamper == "foreground-ancestry": h.processes[102]["ppid"] = 1
    elif tamper == "foreign-cgroup": h.processes[102]["cgroups"] = ["/other.service"]
    elif tamper == "environment": h.processes[100]["environment"] = {}
    elif tamper == "unit-argv": h.view["argv"] = ["/usr/bin/node", "other.js"]
    elif tamper == "fragment": c.unit_fragment_path.write_text("changed")
    elif tamper == "drop-in": h.view["DropInPaths"] = "/unregistered/drop-in"
    elif tamper == "reload": h.view["NeedDaemonReload"] = "yes"
    else:
        pointer = json.loads(c.lease_owner_path.read_text());pointer["lane"]["id"] = "another-owner";write(c.lease_owner_path, pointer)
    result = case.observe(now=2000, apply=True)
    assert result["action"] == "observation_blocked"
    assert h.restarts == 0


def test_pid_birth_race_between_observation_and_action_is_refused(case, monkeypatch):
    original = case.watchdog._scope
    def scope(*args):
        original(*args)
        case.host.processes[102]["start_ticks"] = "new-birth"
    monkeypatch.setattr(case.watchdog, "_scope", scope)
    assert case.observe(now=1030, apply=True)["action"] == "observation_blocked"
    assert case.host.restarts == 0


def test_deadline_is_not_extended_by_heartbeat_logs_native_cpu_or_output_changes(case):
    first = case.observe()
    case.host.processes[104]["cpu_ticks"] = 999
    write(case.config.status_path, {"pid": 102, "active_task_id": "TEST-001", "phase": "deterministic_command", "heartbeat_at_ms": 1030000})
    case.config.log_path.write_text("new heartbeat and streaming output\n")
    write(case.root / "output.json", {"new": "work"})
    second = case.observe(now=1030)
    assert second["action"] == "recovery_due"
    assert first["task_deadline_at"] == second["task_deadline_at"]
    assert case.host.restarts == 0


def test_apply_restarts_only_the_exact_unit_and_budget_survives_watchdog_restart(case):
    first = case.observe(now=1030, apply=True)
    assert first["action"] == "service_restart_requested" and case.host.restarts == 1
    after = case.observe(now=1040, apply=True, restart_object=True)
    assert after["action"] == "recovery_cooldown" and after["cleanup_verified"] is True
    assert case.host.restarts == 1
    case.observe(now=1060, apply=True, restart_object=True)
    case.observe(now=1090, apply=True, restart_object=True)
    exhausted = case.observe(now=1120, apply=True, restart_object=True)
    assert exhausted["reason"] == "persisted_restart_budget_exhausted"
    assert case.host.restarts == 3


def test_hold_never_calls_systemd_and_does_not_change_taskboard(case):
    before = case.config.board_path.read_bytes()
    (case.root / "OPERATOR_STOP").touch()
    assert case.observe(now=2000, apply=True)["action"] == "operator_hold"
    assert case.host.restarts == 0 and case.config.board_path.read_bytes() == before


def test_unowned_dirty_scope_prevents_restart(case):
    case.host.dirt = ["output.json"]
    write(case.root / "output.json", {"owner": "someone else"})
    assert case.observe(now=1030, apply=True)["action"] == "observation_blocked"
    assert case.host.restarts == 0


def test_blocked_frontier_is_actionable_without_restart(case):
    case.config.board_path.write_text(case.config.board_path.read_text().replace("Status: ready", "Status: blocked"))
    result = case.observe(now=2000, apply=True)
    assert result["reason"] == "no_selectable_pending_task"
    assert result["incidents"][0]["code"] == "blocked_ready_frontier"
    assert case.host.restarts == 0


def test_absent_service_waits_grace_and_never_spawns_raw_lane(case):
    case.host.processes.clear();case.host.view.update(MainPID="0", ActiveState="inactive")
    assert case.observe(now=1000, apply=True)["action"] == "startup_grace"
    assert case.observe(now=1011, apply=True)["action"] == "service_restart_requested"
    assert case.host.restarts == 1


def test_unmanaged_matching_foreground_prevents_recovery(case):
    case.host.view.update(MainPID="0", ActiveState="inactive")
    assert case.observe(now=2000, apply=True)["action"] == "observation_blocked"
    assert case.host.restarts == 0


def test_old_task_provider_errors_do_not_label_current_healthy_task(case):
    case.config.log_path.write_text(
        "$ python3 old.py --task-id TEST-999\nTimeoutExpired secret-provider-body\n"
        "$ python3 produce.py --task-id TEST-001  # cwd=/fixture\nordinary work\n")
    result = case.observe()
    assert result["provider_failure_codes"] == []
    assert result["historical_provider_failure_codes"] == ["provider_timeout"]
    assert "secret-provider-body" not in json.dumps(result)
    assert not result["incidents"]


def test_current_native_incomplete_is_an_incident_without_premature_restart(case):
    case.config.log_path.write_text("$ python3 produce.py --task-id TEST-001\nGrok CLI streaming response is incomplete or invalid\n")
    result = case.observe(apply=True)
    assert result["provider_failure_codes"] == ["native_incomplete"]
    assert result["incidents"][0]["code"] == "provider_failures_observed"
    assert case.host.restarts == 0


def test_retry_wait_is_respected_until_its_bounded_deadline(case):
    status = json.loads(case.config.status_path.read_text());status.update(phase="attempt_retry_wait", next_retry_at_ms=1010000)
    write(case.config.status_path, status)
    assert case.observe()["action"] == "bounded_retry_wait"
    assert case.observe(now=1015)["action"] == "bounded_retry_wait"
    assert case.observe(now=1031)["reason"] == "bounded_retry_wait_exceeded"


def test_future_retry_is_not_overridden_by_a_finished_attempt_deadline(case):
    status = json.loads(case.config.status_path.read_text());status.update(phase="attempt_retry_wait", next_retry_at_ms=1500000)
    write(case.config.status_path, status)
    result = case.observe(now=1300, apply=True)
    assert result["task_deadline_remaining_seconds"] < 0
    assert result["action"] == "bounded_retry_wait" and case.host.restarts == 0


def test_extended_retry_timestamp_cannot_postpone_the_bound_indefinitely(case):
    config = replace(case.config, max_backoff_seconds=50)
    monitor = LeasedServiceWatchdog(config, host=case.host)
    status = json.loads(config.status_path.read_text());status.update(phase="attempt_retry_wait", next_retry_at_ms=999999999)
    write(config.status_path, status)
    assert monitor.observe(now=1000)["action"] == "bounded_retry_wait"
    status["next_retry_at_ms"] += 999999999;write(config.status_path, status)
    assert monitor.observe(now=1071)["reason"] == "bounded_retry_wait_exceeded"


def test_structured_native_progress_requires_current_task_and_owned_pid(case):
    config = replace(case.config, progress_globs=("native/*.json",))
    monitor = LeasedServiceWatchdog(config, host=case.host)
    write(case.root / "native/current.json", {"task_id": "TEST-001", "pid": 103, "state": "running", "observed_at_ms": 1000000,
                                              "stream": {"frames": 5, "tool_name_counts": {"read_file": 1}}})
    write(case.root / "native/foreign.json", {"task_id": "TEST-999", "pid": 103, "state": "running", "secret": "provider-body"})
    write(case.root / "native/unowned.json", {"task_id": "TEST-001", "pid": 999, "state": "running"})
    first = monitor.observe(now=1000)
    assert len(first["native_progress"]) == 1
    assert first["native_progress"][0]["read_file_calls"] == 1
    assert first["real_work_observed"] is False
    data = json.loads((case.root / "native/current.json").read_text());data["observed_at_ms"] += 1000
    write(case.root / "native/current.json", data)
    second = monitor.observe(now=1001)
    assert second["native_progress_observed"] is False and second["real_work_observed"] is False
    assert "provider-body" not in json.dumps(second)


def test_new_task_epoch_does_not_reuse_the_previous_task_deadline(case):
    case.observe()
    case.config.board_path.write_text(case.config.board_path.read_text() + "\n## TEST-002 Next\n- Status: ready\n- Depends on:\n- Implementation timeout seconds: 100\n")
    status = json.loads(case.config.status_path.read_text());status.update(active_task_id="TEST-002", last_task_transition_at_ms=2000000)
    write(case.config.status_path, status)
    case.host.processes[103]["argv"][-1] = "TEST-002"
    result = case.observe(now=2010)
    assert result["task_deadline_at"] == 2120


def test_reused_old_pid_does_not_become_a_restart_signal_target(case):
    old = copy.deepcopy(case.host.processes[100]);old.update(argv=["unrelated"], start_ticks="reused", cgroups=["/other.service"])
    case.host.processes = {100: old};case.host.view.update(MainPID="0", ActiveState="inactive")
    assert case.observe(now=1000)["action"] == "startup_grace"
    assert case.observe(now=1011)["action"] == "recovery_due"
    assert case.host.restarts == 0


def test_failed_restart_has_a_durable_bounded_retry_not_an_infinite_cleanup_wait(case, monkeypatch):
    def failed(_config):
        case.host.restarts += 1
        raise ObservationError("unit restart failed")
    monkeypatch.setattr(case.host, "restart", failed)
    assert case.observe(now=1030, apply=True)["action"] == "observation_blocked"
    assert case.observe(now=1040, apply=True, restart_object=True)["action"] == "recovery_cooldown"
    case.observe(now=1060, apply=True, restart_object=True)
    assert case.host.restarts == 2


def test_cli_once_remains_observational_even_with_apply(case, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.rescue import leased_service_watchdog as module
    called = []
    class Monitor:
        def __init__(self, _config): pass
        def observe(self, *, apply):
            called.append(apply);return {"schema": SCHEMA}
    monkeypatch.setattr(module.Config, "from_dict", lambda _value: case.config)
    monkeypatch.setattr(module, "LeasedServiceWatchdog", Monitor)
    source = case.root / "config.json";write(source, {})
    assert module.main(["--config", str(source), "--once", "--apply"]) == 0
    assert called == [False]


@pytest.mark.parametrize("board", ["## TEST-001 X\n- Status: ready\n- Depends on: TEST-002\n", "## TEST-001 X\n- Status: ready\n- Depends on: TEST-001\n"])
def test_taskboard_integrity_fails_closed(board):
    with pytest.raises(ObservationError): parse_board(board, "## TEST-")


def test_configuration_rejects_wildcard_units_unbounded_deadlines_and_output_collisions(case):
    c = case.config
    with pytest.raises(ObservationError): Config(**{**c.__dict__, "unit": "*.service"}).validate()
    with pytest.raises(ObservationError): Config(**{**c.__dict__, "max_task_timeout_seconds": float("inf")}).validate()
    with pytest.raises(ObservationError): Config(**{**c.__dict__, "state_dir": c.board_path.parent}).validate()


def _progress_ledger(case, failures, *, qualified=True):
    from ipfs_accelerate_py.agent_supervisor.rescue.lane_resume import _command_contract
    fields = parse_board(case.config.board_path.read_text(), case.config.task_prefix)["TEST-001"]
    contract = _command_contract(fields, ["python3", "produce.py", "--task-id", "TEST-001"],
                                 ["python3", "validate.py"], repo_root=case.root)
    value = {"TEST-001": failures}
    if qualified:
        value["_validation_fixed_points"] = {"TEST-001": {"execution_progress": {
            "schema": "agent-supervisor/execution-progress@1", "contract": contract,
            "state": "running", "semantic_failures": failures, "attempts_started": failures + 1}}}
    path = case.config.status_path.with_name("lane_resume_attempts.json");write(path, value)
    return path


def test_new_attempt_is_degraded_after_prior_semantic_failure(case):
    _progress_ledger(case, 1)
    result = case.observe()
    assert result["health"] == "degraded"
    assert result["action"] == "active_with_prior_semantic_failure"
    assert case.host.restarts == 0


def test_provider_failure_loop_survives_pid_and_watchdog_resets_without_restart_thrashing(case):
    _progress_ledger(case, 3)
    case.config.log_path.write_text("$ python3 produce.py --task-id TEST-001\nTimeoutExpired\n"
                                   "$ python3 produce.py --task-id TEST-001\nnative progress\n")
    first = case.observe(apply=True)
    assert first["reason"] == "provider_failure_loop"
    fingerprint = next(i["incident_fingerprint"] for i in first["incidents"] if i["code"] == "provider_failure_loop")
    case.host.install_owner(200)
    second = case.observe(now=1001, apply=True, restart_object=True)
    assert second["reason"] == "provider_failure_loop"
    assert next(i["incident_fingerprint"] for i in second["incidents"] if i["code"] == "provider_failure_loop") == fingerprint
    assert case.host.restarts == 0


def test_required_exhausted_barrier_is_not_healthy_even_with_fresh_native_work(case):
    _progress_ledger(case, 6)
    case.config.board_path.write_text(case.config.board_path.read_text() + "\n## TEST-002 Verify\n- Status: ready\n- Depends on: TEST-001\n")
    result = case.observe(apply=True)
    assert result["reason"] == "exhausted_required_barrier"
    incident = next(item for item in result["incidents"] if item["code"] == "exhausted_required_barrier")
    assert incident["blocked_task_ids"] == ["TEST-002"]
    assert result["health"] == "degraded" and case.host.restarts == 0


def _foreground_recovery(case, *, state="running", started=990000, timeout=60):
    from ipfs_accelerate_py.agent_supervisor.rescue import semantic_recovery
    path = _progress_ledger(case, 6)
    contract = json.loads(path.read_text())["_validation_fixed_points"]["TEST-001"]["execution_progress"]["contract"]
    journal = semantic_recovery.Journal(semantic_recovery.journal_path(path), lambda target, content: target.write_text(content))
    row = journal.admit("TEST-001", contract=contract, fingerprint="a" * 64, limit=1,
                        context_path="runtime/repair.context.json", now_ms=started)
    row.update(state=state, timeout_seconds=timeout, recovery_limit=1)
    if state != "running":
        row.update(blocked_contract=contract, reason="recovery_no_material_production_source_change")
    journal.save()
    status = json.loads(case.config.status_path.read_text())
    status["phase"] = "semantic_recovery" if state == "running" else "semantic_recovery_" + state
    write(case.config.status_path, status)
    return journal


def test_foreground_repair_uses_its_admitted_finite_deadline_despite_old_producer_anchor(case):
    case.observe(now=1000)
    _foreground_recovery(case)
    result = case.observe(now=1050, apply=True)
    assert result["task_deadline_at"] == 1070
    assert result["foreground_semantic_recovery"]["spent"] == 1
    assert result["health"] == "degraded" and case.host.restarts == 0
    case.host.processes[103]["start_wall"] = 1059
    case.host.processes[104]["start_wall"] = 1059
    again = case.observe(now=1060, apply=True, restart_object=True)
    assert again["task_deadline_at"] == 1070 and case.host.restarts == 0


def test_exhausted_semantics_do_not_mask_overdue_foreground_repair_recovery(case):
    _foreground_recovery(case)
    result = case.observe(now=1071, apply=True)
    assert result["action"] == "service_restart_requested"
    assert result["reason"] == "declared_task_deadline_exceeded"
    assert case.host.restarts == 1


def test_completed_noop_repair_is_a_hard_block_not_an_old_worker_deadline(case):
    _foreground_recovery(case, state="noop")
    case.host.processes.pop(103);case.host.processes.pop(104)
    result = case.observe(now=2000, apply=True)
    assert result["action"] == "attention_required"
    assert result["reason"] == "bounded_semantic_recovery_exhausted"
    assert result["foreground_semantic_recovery"]["spent"] == 1
    assert case.host.restarts == 0


def test_declared_one_failure_canary_barrier_is_exhausted_after_its_actual_first_failure(case):
    case.config.board_path.write_text(case.config.board_path.read_text() + "- Semantic failure limit: 1\n"
        "\n## TEST-002 Dependent\n- Status: ready\n- Depends on: TEST-001\n")
    _progress_ledger(case, 1)
    result = case.observe(apply=True)
    assert result["reason"] == "exhausted_required_barrier"
    assert result["health"] == "degraded" and case.host.restarts == 0


def test_invalid_recovery_journal_does_not_mask_verified_physical_owner_absence(case):
    from ipfs_accelerate_py.agent_supervisor.rescue import semantic_recovery
    path = _progress_ledger(case, 6)
    semantic_recovery.journal_path(path).write_text('{"invalid":"journal"}')
    case.host.processes.clear();case.host.view.update(MainPID="0", ActiveState="inactive")
    first = case.observe(now=1000, apply=True)
    assert first["action"] == "attention_required" and case.host.restarts == 0
    second = case.observe(now=1011, apply=True)
    assert second["action"] == "service_restart_requested"
    assert second["reason"] == "healthy_foreground_owner_absent" and case.host.restarts == 1


def test_approved_repair_grants_one_persisted_original_probe_anchor_after_a_long_hold(case):
    case.observe(now=1000)
    journal = _foreground_recovery(case, state="applied")
    row = journal.data["tasks"]["TEST-001"]["history"][-1]
    row["approved_contract"] = row.pop("blocked_contract")
    row["finished_at_ms"] = 1000000
    journal.save()
    path = case.config.status_path.with_name("lane_resume_attempts.json")
    ledger = json.loads(path.read_text())
    record = ledger["_validation_fixed_points"]["TEST-001"]["execution_progress"]
    record.update(semantic_failures=0, state="running", started_at_ms=2000000)
    write(path, ledger)
    status = json.loads(case.config.status_path.read_text());status["phase"] = "deterministic_command"
    status["heartbeat_at_ms"] = 2000000;write(case.config.status_path, status)
    case.host.processes[103]["start_wall"] = 2000
    result = case.observe(now=2001, apply=True)
    assert result["task_deadline_at"] == 2120 and case.host.restarts == 0
    record["started_at_ms"] = 2120000;write(path, ledger)
    case.host.processes[103]["start_wall"] = 2120
    late = case.observe(now=2121, apply=True, restart_object=True)
    assert late["task_deadline_at"] == 2120
    assert late["reason"] == "declared_task_deadline_exceeded" and case.host.restarts == 1


def test_legacy_failures_with_inflight_proof_removed_are_explicitly_unqualified_not_healthy(case):
    _progress_ledger(case, 35, qualified=False)
    result = case.observe(apply=True)
    assert result["health"] == "degraded" and result["action"] == "attention_required"
    assert any(i.get("failure_count") == 35 and i.get("contract_qualified") is False for i in result["incidents"])
    assert case.host.restarts == 0


def test_semantic_stall_anchor_is_not_renewed_by_new_service_or_worker_pids(case):
    case.observe()
    case.host.now = 2900;case.host.install_owner(200)
    result = case.observe(now=2900, apply=True, restart_object=True)
    assert result["reason"] == "sustained_no_semantic_progress"
    assert case.host.restarts == 0


def test_only_actual_contract_repair_renews_failure_loop_eligibility(case):
    _progress_ledger(case, 3)
    assert case.observe()["action"] == "attention_required"
    producer = case.root / "produce.py"
    producer.write_text("# actual new implementation\n")
    result = case.observe(now=1001, apply=True, restart_object=True)
    assert result["action"] == "active_with_prior_semantic_failure"
    assert result["health"] == "degraded" and result["unresolved_prior_failure"] is True
    persisted = json.loads(case.watchdog.state_path.read_text())
    assert persisted["task_progress"]["TEST-001"]["observed_failures"] == 0
    assert not any(i.get("code") == "repeated_semantic_failure_loop" for i in result["incidents"])
    assert case.host.restarts == 0


def test_changed_contract_zero_failure_attempts_retain_history_health_until_validation(case):
    path = _progress_ledger(case, 0)
    ledger = json.loads(path.read_text())
    task = ledger["_validation_fixed_points"]["TEST-001"]
    task["execution_progress_history"] = [{"contract": "old-failed", "state": "failed", "semantic_failures": 6}]
    write(path, ledger)
    result = case.observe(apply=True)
    assert result["health"] == "degraded" and result["action"] == "active_with_prior_semantic_failure"
    assert result["unresolved_prior_failure_task_ids"] == ["TEST-001"]
    assert result["unresolved_prior_failure_reason"] == "prior_semantic_failure_not_validated"
    assert not any(i["code"].startswith("exhausted_") for i in result["incidents"])
    assert case.host.restarts == 0
    task["execution_progress"].update(state="validated", semantic_failures=0, validated_at_ms=1000001,
                                      unresolved_prior_failure=False)
    write(path, ledger)
    result = case.observe(now=1001, apply=True, restart_object=True)
    assert result["health"] == "operational" and result["action"] == "active_within_deadline"
    assert result["unresolved_prior_failure"] is False
    assert "unresolved_prior_failure_reason" not in result
    assert case.host.restarts == 0


def test_unresolved_old_contract_health_does_not_suppress_physical_deadline(case):
    path = _progress_ledger(case, 0)
    ledger = json.loads(path.read_text())
    ledger["_validation_fixed_points"]["TEST-001"]["execution_progress_history"] = [
        {"contract": "old-failed", "state": "failed", "semantic_failures": 6}]
    write(path, ledger)
    result = case.observe(now=1030, apply=True)
    assert result["health"] == "degraded" and result["unresolved_prior_failure"] is True
    assert result["action"] == "service_restart_requested"
    assert result["reason"] == "declared_task_deadline_exceeded" and case.host.restarts == 1


def test_semantic_failure_does_not_suppress_verified_owner_absence_recovery(case):
    _progress_ledger(case, 6)
    case.host.processes.clear();case.host.view.update(MainPID="0", ActiveState="inactive")
    first = case.observe(now=1000, apply=True)
    assert first["action"] == "attention_required" and case.host.restarts == 0
    second = case.observe(now=1011, apply=True)
    assert second["action"] == "service_restart_requested"
    assert second["reason"] == "healthy_foreground_owner_absent"
    assert case.host.restarts == 1


def test_unrelated_barrier_does_not_suppress_the_active_task_hard_deadline(case):
    from ipfs_accelerate_py.agent_supervisor.rescue.lane_resume import _command_contract
    case.config.board_path.write_text(case.config.board_path.read_text() +
        "\n## TEST-002 Barrier\n- Status: ready\n- Depends on:\n- Deterministic command: python3 other.py\n"
        "\n## TEST-003 Verify\n- Status: ready\n- Depends on: TEST-002\n")
    fields = parse_board(case.config.board_path.read_text(), case.config.task_prefix)["TEST-002"]
    contract = _command_contract(fields, ["python3", "other.py"], None, repo_root=case.root)
    write(case.config.status_path.with_name("lane_resume_attempts.json"), {
        "TEST-002": 6, "_validation_fixed_points": {"TEST-002": {"execution_progress": {
            "contract": contract, "semantic_failures": 6, "state": "failed"}}}})
    result = case.observe(now=1030, apply=True)
    assert result["action"] == "service_restart_requested"
    assert result["reason"] == "declared_task_deadline_exceeded"
    assert any(i["code"] == "exhausted_required_barrier" for i in result["incidents"])
    assert case.host.restarts == 1


def test_semantic_wait_without_a_producer_is_not_restarted_for_an_old_deadline(case):
    _progress_ledger(case, 6)
    status = json.loads(case.config.status_path.read_text());status["phase"] = "semantic_failure_wait"
    write(case.config.status_path, status)
    case.host.processes.pop(103);case.host.processes.pop(104)
    result = case.observe(now=2000, apply=True)
    assert result["action"] == "attention_required" and case.host.restarts == 0


def test_empty_active_task_backoff_is_not_misreported_as_idle_stall(case):
    status = json.loads(case.config.status_path.read_text())
    status.update(active_task_id="", phase="attempt_retry_wait", next_retry_at_ms=1500000)
    write(case.config.status_path, status)
    assert case.observe(now=1000)["action"] == "bounded_retry_wait"
    assert case.observe(now=1400, apply=True)["action"] == "bounded_retry_wait"
    assert case.host.restarts == 0
