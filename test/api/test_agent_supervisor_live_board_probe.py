"""Regression tests for live board observations, independent of the ML package."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SOURCE = Path(__file__).resolve().parents[2] / "ipfs_accelerate_py/agent_supervisor/rescue/live_board_probe.py"
SPEC = importlib.util.spec_from_file_location("live_board_probe_under_test", SOURCE)
assert SPEC and SPEC.loader
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def board(tmp_path, monkeypatch):
    value = {"id": "pcpr", "cwd": str(tmp_path), "state_root": str(tmp_path / "state"),
             "owner_status_path": str(tmp_path / "owner.json"), "max_lanes": 1,
             "quack_endpoint": "quack:127.0.0.1:1234", "status_argv": ["unused"],
             "ensure_argv": ["native-ensure"]}
    birth = {"pid": 50, "boot_id": "boot", "start_time_ticks": 20}
    _write(Path(value["owner_status_path"]), {"identity": {"process_birth": birth}, "lifecycle": "ready"})
    identities = {50: {**birth, "cwd": str(tmp_path)}}
    monkeypatch.setattr(probe, "process_identity", lambda pid: identities.get(pid, {}))
    monkeypatch.setattr(probe, "_lane_process", lambda *args: {"pid": args[0]} if args[0] else {})
    monkeypatch.setattr(probe, "_source_heads", lambda _: {".": "a" * 40})
    monkeypatch.setattr(probe, "_provider_busy", lambda *_: [])

    class Connected:
        def __enter__(self): return self
        def __exit__(self, *_): pass
    monkeypatch.setattr(probe.socket, "create_connection", lambda *a, **kw: Connected())
    lane = tmp_path / "state/lane-0"
    _write(lane / "pcpr_lane_0_supervisor_status.json", {"supervisor_pid": 60, "daemon_pid": 61,
        "updated_at": "1970-01-01T00:16:40+00:00", "status": "running"})
    return value, identities, lane


def test_no_offline_status_or_completion_when_owner_is_dead(board, monkeypatch):
    config, identities, _ = board
    identities.clear()
    monkeypatch.setattr(probe, "_status_command", lambda _: pytest.fail("must not open offline authority"))
    result = probe.observe_board(config, now=1000)
    assert result["health"] == "stopped"
    assert result["recovery_action"] == "ensure"
    assert result["complete"] is False


def test_reused_owner_pid_is_not_live(board, monkeypatch):
    config, identities, _ = board
    identities[50]["start_time_ticks"] = 21
    monkeypatch.setattr(probe, "_status_command", lambda _: pytest.fail("PID reuse is not a live owner"))
    assert probe.observe_board(config, now=1000)["health"] == "stopped"


def test_stale_checkpoint_never_authorizes_completion(board, monkeypatch):
    config, _, lane = board
    monkeypatch.setattr(probe, "_status_command", lambda _: ({
        "ready": True, "task_authority": {"status_counts": {"complete": 67}, "task_count": 67,
        "transport": "quack_owner_checkpointed_read_replica", "authoritative": False}}, ""))
    _write(lane / "pcpr_lane_0_task_state.json", {"heartbeat_at": "1970-01-01T00:16:35+00:00",
        "projection_complete": True, "task_count": 67, "task_statuses": {"PCPR-001": "blocked",
        "PCPR-002": "todo"}})
    result = probe.observe_board(config, now=1000)
    assert result["health"] == "blocked"
    assert result["details"]["task_counts"] == {"blocked": 1, "todo": 1}
    assert result["complete"] is False
    assert result["completion_candidate"] is False


def test_all_tasks_complete_only_proposes_separate_gate(board, monkeypatch):
    config, _, _ = board
    monkeypatch.setattr(probe, "_status_command", lambda _: ({
        "task_authority": {"status_counts": {"completed": 67}, "task_count": 67,
        "authenticated_query": True}}, ""))
    result = probe.observe_board(config, now=1000)
    assert result["completion_candidate"] is True
    assert result["complete"] is False


def test_progress_token_excludes_heartbeats_and_projection_refreshes():
    before = {"task_statuses": {"T-001": "todo"}, "source_revision": 1,
              "heartbeat_at": "yesterday", "last_progress_at": "yesterday", "source_projection_cid": "old"}
    after = {**before, "source_revision": 20, "heartbeat_at": "now",
             "last_progress_at": "now", "source_projection_cid": "new"}
    assert probe._progress(before) == probe._progress(after)
    assert probe._progress(after) != probe._progress({**after, "task_statuses": {"T-001": "completed"}})


def test_daemon_pid_alone_is_never_provider_activity(board, monkeypatch):
    config, _, _ = board
    monkeypatch.setattr(probe, "_status_command", lambda _: ({"ready": True,
        "task_authority": {"status_counts": {"todo": 67}, "task_count": 67}}, ""))
    result = probe.observe_board(config, now=1000)
    assert result["details"]["lanes"][0]["daemon"]["pid"] == 61
    assert result["busy"] is False


def test_zombie_process_is_rejected(tmp_path):
    proc = tmp_path / "100"
    proc.mkdir()
    (proc / "stat").write_text("100 (python worker) Z 1 " + "0 " * 22)
    assert probe.process_identity(100, proc_root=tmp_path) == {}


def test_hardened_cwd_does_not_hide_birth_identity(tmp_path, monkeypatch):
    proc = tmp_path / "100"
    proc.mkdir()
    fields = ["S", "1"] + ["0"] * 17 + ["42"] + ["0"] * 3
    (proc / "stat").write_text("100 (python worker) " + " ".join(fields))
    (proc / "cmdline").write_bytes(b"python3\0-m\0safe_module\0")
    boot = tmp_path / "sys/kernel/random/boot_id"
    boot.parent.mkdir(parents=True)
    boot.write_text("known-boot")
    original = Path.resolve
    def resolve(path, *args, **kwargs):
        if path == proc / "cwd":
            raise PermissionError("nondumpable")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "resolve", resolve)
    observed = probe.process_identity(100, proc_root=tmp_path)
    assert observed["cwd"] == ""
    assert probe.birth_matches(observed, {"pid": 100, "start_time_ticks": 42, "boot_id": "known-boot"})


def test_cancelled_task_is_not_complete(board, monkeypatch):
    config, _, _ = board
    monkeypatch.setattr(probe, "_status_command", lambda _: ({"task_authority": {
        "status_counts": {"completed": 66, "cancelled": 1}, "task_count": 67}}, ""))
    assert probe.observe_board(config, now=1000)["completion_candidate"] is False


def test_timeout_kills_descendants_after_leader_exits(monkeypatch, tmp_path):
    class Child:
        pid = 500
        waits = 0
        def wait(self, timeout):
            self.waits += 1
            if self.waits == 1:
                raise probe.subprocess.TimeoutExpired("native-status", timeout)
            return 0
    child = Child()
    signals = []
    monkeypatch.setattr(probe.subprocess, "Popen", lambda *a, **kw: child)
    monkeypatch.setattr(probe.os, "killpg", lambda pid, signum: signals.append((pid, signum)))
    result, reason = probe._status_command({"status_argv": ["native-status"], "cwd": str(tmp_path)})
    assert result == {}
    assert reason == "native_status_timeout"
    assert signals == [(500, probe.signal.SIGTERM), (500, probe.signal.SIGKILL)]


def test_hashing_gap_within_ten_minutes_is_not_stale(board, monkeypatch):
    config, _, lane = board
    monkeypatch.setattr(probe, "_status_command", lambda _: ({"task_authority": {
        "status_counts": {"todo": 67}, "task_count": 67}}, ""))
    status_path = lane / "pcpr_lane_0_supervisor_status.json"
    status = json.loads(status_path.read_text())
    status["updated_at"] = "1970-01-01T00:11:40+00:00"
    _write(status_path, status)
    result = probe.observe_board(config, now=1000)
    assert "lane_0_supervisor_heartbeat_stale" not in result["reason_codes"]
    config["supervisor_heartbeat_stale_seconds"] = 150
    assert "lane_0_supervisor_heartbeat_stale" in probe.observe_board(config, now=1000)["reason_codes"]


@pytest.mark.parametrize("argv", [
    ["python3", "-m", "ipfs_accelerate_py.agent_supervisor.runtime.grok_cli_runner"],
    ["python3", "-m", "ipfs_accelerate_py.agent_supervisor.runtime.provider_fallback_runner"],
    ["/home/user/.local/bin/grok-1.0.24-linux-aarch64"],
    ["/usr/local/lib/node_modules/@openai/codex/vendor/aarch64-unknown-linux-gnu/codex/codex"],
])
def test_packaged_provider_command_names(argv):
    assert probe._provider_command(argv)


def test_prompt_mention_of_provider_is_not_a_provider_process():
    assert not probe._provider_command(["python3", "-c", "text mentioning grok_cli_runner"])


def test_native_status_uses_board_package_and_keeps_capability_environment(monkeypatch, tmp_path):
    external = tmp_path / "external/ipfs_accelerate"
    external.mkdir(parents=True)
    monkeypatch.setenv("PYTHONPATH", "/released-fleet-stub")
    monkeypatch.setenv("IPFS_HASH_MAX_WORKERS", "1")
    observed = {}
    class Child:
        def wait(self, timeout): return 0
    def start(argv, **kwargs):
        observed.update(kwargs["env"])
        kwargs["stdout"].write(b'{"ready": true}')
        return Child()
    monkeypatch.setattr(probe.subprocess, "Popen", start)
    result, error = probe._status_command({"status_argv": ["native-status"], "cwd": str(tmp_path)})
    assert result == {"ready": True}
    assert error == ""
    assert observed["PYTHONPATH"] == str(tmp_path) + probe.os.pathsep + str(external)
    assert observed["IPFS_HASH_MAX_WORKERS"] == "1"


def _aseh_native_receipt(*, cursor=10, status="in_progress"):
    authority = {"available": True, "transport": "quack",
                 "credential_path": "sealed_memfd_broker",
                 "snapshot": {"task_count": 2}, "event_cursor": cursor,
                 "task_statuses": {"ASEH-001": "completed", "ASEH-061": status}}
    return {"healthy": True, "broker_authenticated_receipt": True,
            "receipt": {"broker_authenticated": True,
                        "samples": [{"authority": dict(authority)},
                                    {"authority": dict(authority)}]}}


def test_aseh_extracts_progress_from_admitted_nested_samples(board, monkeypatch):
    config, _, _ = board
    config = {**config, "board_id": "aseh"}
    native = _aseh_native_receipt()
    monkeypatch.setattr(probe, "_status_command", lambda _: (native, ""))
    before = probe.observe_board(config, now=1000)
    assert before["details"]["task_counts"] == {"completed": 1, "in_progress": 1}
    assert before["details"]["task_count"] == 2
    assert before["details"]["event_cursor"] == 10
    assert before["details"]["authenticated_task_observation"] is True
    native["receipt"]["observed_at"] = 1234
    assert probe.observe_board(config, now=1000)["progress_token"] == before["progress_token"]
    native = _aseh_native_receipt(cursor=11, status="completed")
    after = probe.observe_board(config, now=1000)
    assert after["progress_token"] != before["progress_token"]
    assert after["completion_candidate"] is True
    assert after["complete"] is False


@pytest.mark.parametrize("case", ["rejected", "unbound", "missing", "unavailable", "wrong_transport"])
def test_aseh_rejects_unadmitted_nested_samples(board, monkeypatch, case):
    config, _, _ = board
    config = {**config, "board_id": "aseh"}
    native = _aseh_native_receipt(status="completed")
    if case == "rejected":
        native["broker_authenticated_receipt"] = False
    elif case == "unbound":
        native["receipt"]["broker_authenticated"] = False
    elif case == "missing":
        native["receipt"]["samples"] = []
    elif case == "unavailable":
        native["receipt"]["samples"][0]["authority"]["available"] = False
    else:
        native["receipt"]["samples"][-1]["authority"]["transport"] = "read_replica"
    monkeypatch.setattr(probe, "_status_command", lambda _: (native, ""))
    result = probe.observe_board(config, now=1000)
    assert result["details"]["authenticated_task_observation"] is False
    assert result["details"]["task_counts"] == {}
    assert result["progress_token"] == ""
    assert result["complete"] is False
    assert result["completion_candidate"] is False


@pytest.mark.parametrize("field", ["blocked_task_ids", "failed_or_blocked_task_ids"])
def test_native_blocker_ids_survive_probe_and_prevent_completion(board, monkeypatch, field):
    config, _, _ = board
    authority = {"status_counts": {"completed": 85}, "task_count": 85,
                 "authenticated_query": True, field: ["DOEP-011", "DOEP-031", "DOEP-072"]}
    monkeypatch.setattr(probe, "_status_command", lambda _: ({"task_authority": authority}, ""))
    result = probe.observe_board(config, now=1000)
    assert result["details"]["blocked_task_ids"] == ["DOEP-011", "DOEP-031", "DOEP-072"]
    assert result["health"] == "blocked"
    assert result["completion_candidate"] is False
    assert result["complete"] is False


def test_native_blocker_identity_changes_progress_without_count_changes():
    before = {"status_counts": {"blocked": 1}, "task_count": 1,
              "failed_or_blocked_task_ids": ["DOEP-011"]}
    after = {**before, "failed_or_blocked_task_ids": ["DOEP-031"]}
    assert probe._progress(before) != probe._progress(after)
    assert probe._progress(before) == probe._progress({
        **before, "failed_or_blocked_task_ids": ["DOEP-011", "DOEP-011"],
        "blocked_task_ids": ["DOEP-011"], "updated_at": "later"})


@pytest.mark.parametrize("invalid", [None, "DOEP-011", {"DOEP-011": True}, [None, 7, {}, ""]])
def test_malformed_native_blocker_ids_are_not_task_identities(board, monkeypatch, invalid):
    config, _, _ = board
    monkeypatch.setattr(probe, "_status_command", lambda _: ({"task_authority": {
        "status_counts": {"todo": 1}, "task_count": 1,
        "failed_or_blocked_task_ids": invalid, "blocked_task_ids": ["DOEP-031"]}}, ""))
    assert probe.observe_board(config, now=1000)["details"]["blocked_task_ids"] == ["DOEP-031"]


def _unavailable_native_receipt():
    return {"healthy": False, "owner_ready": True,
            "broker_authenticated_receipt": False, "receipt": {},
            "receipt_error": {"error_type": "OperatorError",
                              "reason": "live_status_receipt_unavailable_or_invalid"}}


@pytest.fixture
def receipt_clock(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(probe.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(probe.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    return clock


def test_native_receipt_retry_admits_only_new_native_result(board, monkeypatch, receipt_clock):
    config, identities, _ = board
    config = {**config, "board_id": "aseh"}
    replies = iter([(_unavailable_native_receipt(), ""), (_aseh_native_receipt(cursor=6653), "")])
    calls = []
    def status(config):
        calls.append(config)
        return next(replies)
    monkeypatch.setattr(probe, "_status_command", status)
    result = probe.observe_board(config, now=1000)
    assert result["details"]["event_cursor"] == 6653
    assert result["details"]["authenticated_task_observation"] is True
    assert result["details"]["native_status_attempts"] == 2
    assert result["complete"] is False
    assert len(calls) == 2
    assert 0 < calls[-1]["status_timeout_seconds"] <= 15


def test_native_receipt_retry_is_bounded_and_never_admits_invalid_receipt(board, monkeypatch, receipt_clock):
    config, identities, _ = board
    calls = []
    def status(config):
        calls.append(config)
        return _unavailable_native_receipt(), ""
    monkeypatch.setattr(probe, "_status_command", status)
    result = probe.observe_board({**config, "board_id": "aseh"}, now=1000)
    assert 1 < len(calls) <= 4
    assert receipt_clock[0] <= 20
    assert result["details"]["authenticated_task_observation"] is False
    assert result["details"]["task_counts"] == {}
    assert "native_receipt_unavailable_after_retry" in result["reason_codes"]
    assert result["complete"] is False


@pytest.mark.parametrize("when", ["before_retry", "during_retry"])
def test_native_receipt_retry_refuses_changed_owner_birth(board, monkeypatch, receipt_clock, when):
    config, identities, _ = board
    calls = []
    def status(config):
        calls.append(config)
        if len(calls) == 2:
            identities[50] = {**identities[50], "start_time_ticks": 21}
            return _aseh_native_receipt(status="completed"), ""
        return _unavailable_native_receipt(), ""
    def sleep(seconds):
        receipt_clock[0] += seconds
        if when == "before_retry":
            identities[50] = {**identities[50], "start_time_ticks": 21}
    monkeypatch.setattr(probe.time, "sleep", sleep)
    monkeypatch.setattr(probe, "_status_command", status)
    result = probe.observe_board({**config, "board_id": "aseh"}, now=1000)
    assert len(calls) == (1 if when == "before_retry" else 2)
    assert result["details"]["authenticated_task_observation"] is False
    assert "native_status_owner_changed" in result["reason_codes"]
    assert result["completion_candidate"] is False


@pytest.mark.parametrize("change,error", [
    ({"healthy": True}, ""), ({"blocked": True}, ""), ({"stuck": True}, ""),
    ({"owner_ready": False}, ""), ({"receipt_error": {}}, ""),
    ({"receipt": {"samples": []}}, ""), ({}, "native_status_timeout"),
])
def test_native_receipt_retry_does_not_mask_other_health_decisions(board, monkeypatch, receipt_clock, change, error):
    config, identities, _ = board
    calls = []
    def status(config):
        calls.append(config)
        return {**_unavailable_native_receipt(), **change}, error
    monkeypatch.setattr(probe, "_status_command", status)
    probe.observe_board(config, now=1000)
    assert len(calls) == 1
    assert receipt_clock[0] == 0


def test_native_receipt_retry_timeout_consumes_remaining_window(board, monkeypatch, receipt_clock):
    config, _, _ = board
    calls = []
    def status(config):
        calls.append(config)
        if len(calls) > 1:
            receipt_clock[0] += config["status_timeout_seconds"]
            return {}, "native_status_timeout"
        return _unavailable_native_receipt(), ""
    monkeypatch.setattr(probe, "_status_command", status)
    result = probe.observe_board(config, now=1000)
    assert len(calls) == 2
    assert receipt_clock[0] == 20
    assert "native_status_timeout" in result["reason_codes"]
    assert result["details"]["authenticated_task_observation"] is False


def test_native_receipt_retry_preserves_new_stuck_decision(board, monkeypatch, receipt_clock):
    config, _, _ = board
    stuck = {**_aseh_native_receipt(), "healthy": False, "stuck": True}
    replies = iter([(_unavailable_native_receipt(), ""), (stuck, "")])
    monkeypatch.setattr(probe, "_status_command", lambda _: next(replies))
    result = probe.observe_board({**config, "board_id": "aseh"}, now=1000)
    assert result["health"] == "stalled"
    assert result["details"]["native_status_attempts"] == 2
    assert result["complete"] is False


@pytest.mark.parametrize("log_age,provider_parent,log_lane,expected", [
    (5, 61, 0, True),
    (1201, 61, 0, False),
    (-5, 61, 0, False),
    (5, 999, 0, False),
    (5, 61, 1, False),
])
def test_database_portal_log_requires_recent_output_and_same_lane_provider(
    tmp_path, monkeypatch, log_age, provider_parent, log_lane, expected
):
    lanes = [{"state_dir": str(tmp_path / f"lane-{i}"),
              "daemon": {"pid": 61 + i}} for i in range(2)]
    log = (Path(lanes[log_lane]["state_dir"])
           / f"board_lane_{log_lane}_database_portal_attempts/attempt-binding"
           / "implementation-logs/task-attempt-1.log")
    log.parent.mkdir(parents=True)
    log.write_text("provider tool output")
    probe.os.utime(log, (10000 - log_age, 10000 - log_age))
    identities = {
        61: {"pid": 61, "parent_pid": 1, "argv": ["python3", "daemon"]},
        62: {"pid": 62, "parent_pid": 1, "argv": ["python3", "daemon"]},
        70: {"pid": 70, "parent_pid": provider_parent, "argv": ["python3", "wrapper"]},
        71: {"pid": 71, "parent_pid": 70, "argv": ["grok"]},
    }
    original_iterdir = Path.iterdir
    def iterdir(path):
        if path == Path("/proc"):
            return iter(Path(f"/proc/{pid}") for pid in identities)
        return original_iterdir(path)
    monkeypatch.setattr(Path, "iterdir", iterdir)
    monkeypatch.setattr(probe, "process_identity", lambda pid: identities.get(int(pid), {}))
    result = probe._provider_busy(lanes, {}, now=10000)
    assert [item["pid"] for item in result] == ([71] if expected else [])
