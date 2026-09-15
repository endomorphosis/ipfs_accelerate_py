"""Regression tests for live board observations, independent of the ML package."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


SOURCE = Path(__file__).resolve().parents[2] / "ipfs_accelerate_py/agent_supervisor/rescue/live_board_probe.py"
SPEC = importlib.util.spec_from_file_location("live_board_probe_under_test", SOURCE)
assert SPEC and SPEC.loader
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)
PROVIDER_BUSY = probe._provider_busy


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_json_fifo_cannot_block_probe(tmp_path):
    path = tmp_path / "status.json"
    os.mkfifo(path)
    script = """
import importlib.util, pathlib, sys
spec = importlib.util.spec_from_file_location('probe', sys.argv[1])
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
assert probe.read_json(pathlib.Path(sys.argv[2])) == {}
"""
    # The timeout owns only this disposable reader, even before the repair.
    subprocess.run([sys.executable, "-c", script, str(SOURCE), str(path)],
                   check=True, timeout=3, capture_output=True)


def test_json_symlink_is_not_status_evidence(tmp_path):
    target = tmp_path / "target.json"
    _write(target, {"lifecycle": "ready"})
    path = tmp_path / "status.json"
    path.symlink_to(target)
    assert probe.read_json(path) == {}


@pytest.mark.parametrize("raw", [b"{", b"\xff", b"[]", b"null",
                                   b'{"nested":' + b"[" * 20000 + b"0" + b"]" * 20000 + b"}"],
                         ids=["syntax", "encoding", "array", "null", "recursion"])
def test_json_invalid_or_deep_document_is_unavailable(tmp_path, raw):
    path = tmp_path / "status.json"
    path.write_bytes(raw)
    assert probe.read_json(path) == {}


def test_json_size_bound_includes_exact_limit(tmp_path, monkeypatch):
    path = tmp_path / "status.json"
    raw = b'{"ready":true}'
    monkeypatch.setattr(probe, "MAX_JSON_BYTES", len(raw))
    path.write_bytes(raw)
    assert probe.read_json(path) == {"ready": True}
    path.write_bytes(raw + b" ")
    assert probe.read_json(path) == {}


def test_json_reader_rejects_change_during_read_and_closes_descriptor(tmp_path, monkeypatch):
    path = tmp_path / "status.json"
    original = b'{"ready":true}'
    path.write_bytes(original)
    read = os.read
    observed = []

    def race(fd, size):
        result = read(fd, size)
        if not observed:
            observed.append(fd)
            path.write_bytes(b'{"ready":null}')
        return result

    monkeypatch.setattr(probe.os, "read", race)
    assert probe.read_json(path) == {}
    assert observed
    with pytest.raises(OSError):
        os.fstat(observed[0])


def test_json_read_is_bounded_when_file_grows_after_fstat(tmp_path, monkeypatch):
    path = tmp_path / "status.json"
    path.write_bytes(b"{}")
    monkeypatch.setattr(probe, "MAX_JSON_BYTES", 64)
    read = os.read
    observed = []

    def grow(fd, size):
        if not observed:
            path.write_bytes(b"{}" + b" " * 10000)
        result = read(fd, size)
        observed.append(len(result))
        return result

    monkeypatch.setattr(probe.os, "read", grow)
    assert probe.read_json(path) == {}
    assert observed
    assert sum(observed) <= 65


def test_json_regular_file_can_span_read_chunks(tmp_path):
    path = tmp_path / "status.json"
    value = {"message": "x" * (128 * 1024), "lifecycle": "ready"}
    _write(path, value)
    assert probe.read_json(path) == value


def test_json_directory_and_missing_file_are_unavailable(tmp_path):
    assert probe.read_json(tmp_path) == {}
    assert probe.read_json(tmp_path / "missing.json") == {}


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


def test_missing_checkout_is_typed_not_probe_exception(tmp_path):
    result = probe.observe_board({
        "id": "pcpr", "cwd": str(tmp_path / "gone"), "state_root": str(tmp_path / "gone"),
        "owner_status_path": str(tmp_path / "gone" / "owner.json"), "max_lanes": 1,
    }, now=1000)
    assert result["reason_codes"] == ["board_checkout_missing"]
    assert result["complete"] is False
    assert result["health"] == "unknown"


def test_null_status_identity_uses_exclusive_owner_marker(board, monkeypatch, tmp_path):
    config, identities, _ = board
    marker = tmp_path / ".control.duckdb.state-owner.json"
    birth = identities[50]
    _write(marker, {"process_birth": {
        "pid": birth["pid"], "boot_id": birth["boot_id"],
        "start_time_ticks": birth["start_time_ticks"],
    }})
    _write(Path(config["owner_status_path"]), {
        "identity": None, "lifecycle": "failed",
        "owner_marker_path": str(marker),
    })
    monkeypatch.setattr(probe, "_status_command", lambda _: pytest.fail("not ready is not completion"))
    result = probe.observe_board(config, now=1000)
    assert "owner_status_identity_missing" in result["reason_codes"]
    assert "owner_not_ready" in result["reason_codes"]
    assert "owner_process_missing_or_birth_mismatch" not in result["reason_codes"]
    assert result["details"]["owner_ready"] is False
    assert "recovery_action" not in result


@pytest.mark.parametrize("status", [
    "agentic_maintenance_deferred",
    "agentic_maintenance_started",
    "agentic_maintenance_completed",
])
def test_extra_gate_deferred_lane_is_not_daemon_missing(board, monkeypatch, status):
    config, _, lane = board
    _write(lane / "pcpr_lane_0_supervisor_status.json", {
        "supervisor_pid": 60, "daemon_pid": None, "updated_at": "1970-01-01T00:16:40+00:00",
        "status": status, "stalled_without_active_worker": False,
        "last_exit_code": 1,
    })
    monkeypatch.setattr(probe, "_status_command", lambda _: ({
        "task_authority": {"status_counts": {"in_progress": 1, "todo": 2}, "task_count": 3,
                           "authenticated_query": True},
        "ready": True, "healthy": True, "operational_ready": True,
    }, ""))
    result = probe.observe_board(config, now=1000)
    assert "lane_0_daemon_missing" not in result["reason_codes"]
    assert result["details"]["lanes"][0]["extra_gate_deferred"] is True


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


def test_native_spar_completion_authority_is_the_only_complete_flag(board, monkeypatch):
    config, _, _ = board
    config["id"] = "spar"
    config["board_id"] = "spar"
    payload = {
        "schema": "ipfs_accelerate_py/agent-supervisor/database-board-status@1",
        "authoritative_task_observation": True,
        "completion_authority": False,
        "complete": False,
        "completion_gate": "sealed_goal_and_terminal_receipt_review_required",
    }
    monkeypatch.setattr(
        probe, "_status_with_receipt_retry", lambda *_args, **_kwargs: (payload, "", 1)
    )
    monkeypatch.setattr(
        probe,
        "_database_board_authority",
        lambda *_args, **_kwargs: {
            "task_count": 51,
            "status_counts": {"completed": 51},
            "authenticated_query": True,
        },
    )
    held = probe.observe_board(config, now=1000)
    assert held["complete"] is False
    assert held["details"]["native_completion_authority"] is False
    payload["completion_authority"] = True
    payload["complete"] = True
    payload["completion_gate"] = "native_spar_closeout_profile"
    admitted = probe.observe_board(config, now=1000)
    assert admitted["complete"] is True
    assert admitted["details"]["native_completion_authority"] is True
    assert admitted["details"]["completion_gate"] == "native_spar_closeout_profile"


def test_all_tasks_complete_only_proposes_separate_gate(board, monkeypatch):
    config, _, _ = board
    monkeypatch.setattr(probe, "_status_command", lambda _: ({
        "task_authority": {"status_counts": {"completed": 67}, "task_count": 67,
        "authenticated_query": True}}, ""))
    result = probe.observe_board(config, now=1000)
    assert result["completion_candidate"] is True
    assert result["complete"] is False


def test_native_failure_keeps_closed_diagnostic_without_retry_or_authority(board, monkeypatch, tmp_path):
    config, _, _ = board
    secret = "private-token-must-not-appear"
    native = {"schema": "sawm/operator-error@1", "valid": False, "error": secret,
              "observation_error": {"stage": "sample_not_due", "kind": "unavailable"}}
    script = tmp_path / "failed_status.py"
    script.write_text("import sys\nprint(" + repr(json.dumps(native)) + ")\n"
                      "sys.stderr.write(" + repr(secret * 10000) + ")\nraise SystemExit(2)\n")
    config["status_argv"] = [sys.executable, "-I", "-S", "-B", str(script)]
    original = probe._status_command
    calls = []
    def observed(value):
        calls.append(value)
        return original(value)
    monkeypatch.setattr(probe, "_status_command", observed)
    result = probe.observe_board(config, now=1000)
    assert result["details"]["native_status_diagnostic"] == native["observation_error"]
    assert len(calls) == 1 and result["details"]["native_status_attempts"] == 1
    assert result["details"]["authenticated_task_observation"] is False
    assert result["details"]["task_counts"] == {}
    assert result["complete"] is False and result["completion_candidate"] is False
    assert result["reason_codes"] == ["native_status_nonzero"]
    assert secret not in json.dumps(result) and "recovery_action" not in result


@pytest.mark.parametrize("change", ["unknown_stage", "unknown_kind", "extra", "array", "oversized", "schema", "valid", "zero_exit"])
def test_malformed_or_success_diagnostic_is_omitted_without_changing_health(board, monkeypatch, change):
    config, _, _ = board
    error = "native_status_nonzero"
    native = {"schema": "sawm/operator-error@1", "valid": False,
              "observation_error": {"stage": "peer_receive", "kind": "timeout"}}
    if change == "unknown_stage": native["observation_error"]["stage"] = "secret"
    if change == "unknown_kind": native["observation_error"]["kind"] = "secret"
    if change == "extra": native["observation_error"]["secret"] = "credential"
    if change == "array": native["observation_error"] = ["secret"]
    if change == "oversized": native["observation_error"]["stage"] = "secret" * 10000
    if change == "schema": native["schema"] = "foreign/error"
    if change == "valid": native["valid"] = True
    if change == "zero_exit": error = ""
    monkeypatch.setattr(probe, "_status_command", lambda _: (native, error))
    result = probe.observe_board(config, now=1000)
    assert "native_status_diagnostic" not in result["details"]
    assert result["details"]["authenticated_task_observation"] is False
    assert result["complete"] is False and result["completion_candidate"] is False


def test_progress_token_excludes_heartbeats_and_projection_refreshes():
    before = {"task_statuses": {"T-001": "todo"}, "source_revision": 1,
              "heartbeat_at": "yesterday", "last_progress_at": "yesterday", "source_projection_cid": "old",
              "event_cursor": 10}
    after = {**before, "source_revision": 20, "heartbeat_at": "now",
             "last_progress_at": "now", "source_projection_cid": "new", "event_cursor": 100}
    assert probe._progress(before) == probe._progress(after)
    assert probe._progress(after) != probe._progress({**after, "task_statuses": {"T-001": "completed"}})


def test_progress_requires_task_or_goal_evidence_and_normalizes_identity_sets():
    assert probe._progress({"event_cursor": 100}) == ""
    before = {"active_task_ids": ["T-001", "T-002"],
              "completed_task_ids": ["T-003", "T-004"], "unsettled_goal_count": 2}
    reordered = {**before, "active_task_ids": ["T-002", "T-001", "T-002"],
                 "completed_task_ids": ["T-004", "T-003"]}
    assert probe._progress(before) == probe._progress(reordered)
    assert probe._progress(before) != probe._progress({**before, "active_task_ids": ["T-002"]})
    assert probe._progress(before) != probe._progress({**before, "unsettled_goal_count": 1})


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
    assert observed["process_state"] == "S"
    assert observed["wait_channel"] == ""
    assert probe.birth_matches(observed, {"pid": 100, "start_time_ticks": 42, "boot_id": "known-boot"})


@pytest.mark.parametrize("latest_state,latest_birth,accepted", [
    ("D", "42", True), ("T", "42", True), ("t", "42", True),
    ("Z", "42", False), ("X", "42", False), ("D", "43", False),
])
def test_process_condition_uses_same_birth_final_sample(
    tmp_path, monkeypatch, latest_state, latest_birth, accepted,
):
    proc = tmp_path / "100"
    proc.mkdir()
    (proc / "cmdline").write_bytes(b"python3\0-m\0safe_module\0")
    (proc / "cwd").symlink_to(tmp_path, target_is_directory=True)
    (proc / "wchan").write_text("kernel_clone")
    boot = tmp_path / "sys/kernel/random/boot_id"
    boot.parent.mkdir(parents=True)
    boot.write_text("known-boot")
    def stat(state, birth):
        return "100 (python worker) " + " ".join([state, "1"] + ["0"] * 17 + [birth] + ["0"] * 3)
    samples = iter([stat("S", "42"), stat(latest_state, latest_birth)])
    original = Path.read_text
    def read(path, *args, **kwargs):
        return next(samples) if path == proc / "stat" else original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", read)
    observed = probe.process_identity(100, proc_root=tmp_path)
    if not accepted:
        assert observed == {}
        return
    public = probe._public_identity(observed)
    assert public["process_state"] == latest_state
    assert public["wait_channel"] == "kernel_clone"
    assert public["start_time_ticks"] == 42
    assert "argv" not in public


@pytest.mark.parametrize("role", ["owner", "supervisor", "daemon"])
@pytest.mark.parametrize("state,condition", [("T", "stopped"), ("t", "stopped"), ("D", "uninterruptible")])
def test_fresh_heartbeat_does_not_hide_kernel_process_condition(board, monkeypatch, role, state, condition):
    config, identities, _ = board
    identities[50]["process_state"] = state if role == "owner" else "S"
    affected = {"supervisor": 60, "daemon": 61}.get(role)
    monkeypatch.setattr(probe, "_lane_process", lambda pid, *_: {
        "pid": pid, "process_state": state if pid == affected else "S",
        "start_time_ticks": 42, "boot_id": "known-boot", "wait_channel": "kernel_clone"})
    monkeypatch.setattr(probe, "_status_command", lambda _: ({"task_authority": {
        "status_counts": {"todo": 1}, "task_count": 1, "authenticated_query": True}}, ""))
    # Work in another lane remains visible and must be preserved by recovery.
    monkeypatch.setattr(probe, "_provider_busy", lambda *_: [{"pid": 70}])
    result = probe.observe_board(config, now=1000)
    prefix = "owner" if role == "owner" else f"lane_0_{role}"
    assert f"{prefix}_process_{condition}" in result["reason_codes"]
    assert not any("missing" in reason for reason in result["reason_codes"])
    assert result["health"] == "degraded"
    assert result["busy"] is True
    assert result["complete"] is False
    assert "recovery_action" not in result
    if affected:
        assert result["details"]["lanes"][0][role]["pid"] == affected


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


@pytest.mark.parametrize("log_age,provider_parent,log_lane,expected,scan_seconds", [
    (5, 61, 0, True, 0),
    (1201, 61, 0, False, 0),
    (-5, 61, 0, False, 0),
    (5, 999, 0, False, 0),
    (5, 61, 1, False, 0),
    (-5, 61, 0, True, 10),
    (1195, 61, 0, False, 10),
    (-15, 61, 0, False, 10),
])
def test_database_portal_log_requires_recent_output_and_same_lane_provider(
    tmp_path, monkeypatch, log_age, provider_parent, log_lane, expected, scan_seconds
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
    clock = [0.0]
    monkeypatch.setattr(probe.time, "monotonic", lambda: clock[0])
    original_iterdir = Path.iterdir
    def iterdir(path):
        if path == Path("/proc"):
            clock[0] = scan_seconds
            return iter(Path(f"/proc/{pid}") for pid in identities)
        return original_iterdir(path)
    monkeypatch.setattr(Path, "iterdir", iterdir)
    monkeypatch.setattr(probe, "process_identity", lambda pid: identities.get(int(pid), {}))
    result = probe._provider_busy(lanes, {}, now=10000)
    assert [item["pid"] for item in result] == ([71] if expected else [])


def test_provider_output_during_native_receipt_wait_is_recent(board, monkeypatch):
    config, identities, lane = board
    clock = [0.0]
    monkeypatch.setattr(probe.time, "monotonic", lambda: clock[0])
    log = lane / "implementation-logs/task-attempt-1.log"
    log.parent.mkdir()
    log.write_text("worker output while status waits")
    probe.os.utime(log, (1010, 1010))
    identities.update({
        61: {"pid": 61, "parent_pid": 1, "argv": ["python3", "daemon"]},
        71: {"pid": 71, "parent_pid": 61, "argv": ["grok"]},
    })
    original_iterdir = Path.iterdir
    monkeypatch.setattr(Path, "iterdir", lambda path: (
        iter(Path(f"/proc/{pid}") for pid in (61, 71))
        if path == Path("/proc") else original_iterdir(path)))
    monkeypatch.setattr(probe, "process_identity", lambda pid: identities.get(int(pid), {}))
    monkeypatch.setattr(probe, "_provider_busy", PROVIDER_BUSY)

    def slow_status(_):
        clock[0] = 15.0
        return {"ready": True, "task_authority": {"authenticated_query": True,
                "task_statuses": {"T-001": "in_progress"}}}, ""

    monkeypatch.setattr(probe, "_status_command", slow_status)
    result = probe.observe_board(config, now=1000)
    assert result["busy"] is True
    assert [p["pid"] for p in result["details"]["providers"]] == [71]
    assert result["complete"] is False


def _database_native_status(config, *, goal_status='completed'):
    owner_status = probe.read_json(Path(config['owner_status_path']))
    owner = {**owner_status['identity'], 'server_id': 'server:native',
             'process_birth_id': 'birth:native', 'database_uuid': 'uuid:native',
             'store_id': 'store:native', 'generation': 2, 'fence_epoch': 2}
    _write(Path(config['owner_status_path']), {**owner_status, 'identity': owner})
    config['task_namespace'] = 'board:native'
    state = {'task_cid': 'cid:task', 'status': 'completed', 'revision': 3}
    return {'schema': 'ipfs_accelerate_py/agent-supervisor/database-board-status@1',
            'authoritative_task_observation': True, 'board_namespace': 'board:native',
            'observed_at': 1000, 'owner_identity': owner,
            'tasks': [{**state, 'task_alias': 'PCPR-096'}],
            'control': {'task_count': 1, 'goal_count': 1, 'event_watermark': 7,
                        'goals_json': json.dumps([{'goal_cid': 'goal:native', 'status': goal_status}])},
            'completion_snapshot': {'schema': 'ipfs_accelerate_py/agent-supervisor/typed-completion-progress-snapshot@1',
                'owner_identity': owner, 'completion_projection': {
                    'task_states': [state], 'completion_receipts': [{'receipt_cid': 'cid:receipt'}]}}}


def test_database_native_status_overrides_daemon_projection_without_closing_board(board, monkeypatch):
    config, _, lane = board
    native = _database_native_status(config)
    monkeypatch.setattr(probe, '_status_command', lambda _: (native, ''))
    _write(lane / 'pcpr_lane_0_task_state.json', {'heartbeat_at': 1000,
        'projection_complete': True, 'task_count': 1, 'task_statuses': {'PCPR-096': 'blocked'}})
    result = probe.observe_board(config, now=1000)
    assert result['details']['authenticated_task_observation'] is True
    assert result['details']['task_counts'] == {'completed': 1}
    assert result['details']['completion_receipt_count'] == 1
    assert result['completion_candidate'] is True
    assert result['complete'] is False


def test_database_terminal_tasks_do_not_hide_unsettled_goals(board, monkeypatch):
    config, _, _ = board
    native = _database_native_status(config, goal_status='active')
    monkeypatch.setattr(probe, '_status_command', lambda _: (native, ''))
    result = probe.observe_board(config, now=1000)
    assert result['health'] == 'blocked'
    assert result['details']['unsettled_goal_count'] == 1
    assert 'board_has_unsettled_goals' in result['reason_codes']
    assert result['completion_candidate'] is False
    assert result['complete'] is False


def _stopped_terminal_lane(board, monkeypatch, *, goal_status="active"):
    config, _, lane = board
    native = _database_native_status(config, goal_status=goal_status)
    monkeypatch.setattr(probe, "_status_command", lambda _: (native, ""))
    monkeypatch.setattr(probe, "_owner_writer_custody", lambda *_: {
        "configured": True, "verified": True, "held": True,
    })
    status = {"supervisor_pid": None, "daemon_pid": None, "status": "stopped",
              "updated_at": 1, "last_exit_code": 143,
              "last_recycle_reason": "supervisor_signal_shutdown"}
    _write(lane / "pcpr_lane_0_supervisor_status.json", status)
    return config, lane, native, status


@pytest.mark.parametrize("goal_status", ["active", "completed"])
def test_successful_task_frontier_recognizes_stopped_lanes_without_closing_goals(
    board, monkeypatch, goal_status,
):
    config, _, _, _ = _stopped_terminal_lane(board, monkeypatch, goal_status=goal_status)
    result = probe.observe_board(config, now=1000)
    assert result["details"]["implementation_frontier_complete"] is True
    assert result["details"]["expected_stopped_lanes"] == [0]
    assert not any("missing" in reason for reason in result["reason_codes"])
    assert result["health"] == ("blocked" if goal_status == "active" else "healthy")
    assert result["completion_candidate"] is (goal_status == "completed")
    assert result["complete"] is False
    assert "recovery_action" not in result
    if goal_status == "active":
        assert "board_has_unsettled_goals" in result["reason_codes"]


@pytest.mark.parametrize("status", ["todo", "in_progress", "blocked", "quarantined", "failed", "cancelled"])
def test_stopped_lanes_still_require_repair_for_unsuccessful_tasks(board, monkeypatch, status):
    config, _, native, _ = _stopped_terminal_lane(board, monkeypatch)
    native["tasks"][0]["status"] = status
    native["completion_snapshot"]["completion_projection"]["task_states"][0]["status"] = status
    result = probe.observe_board(config, now=1000)
    assert result["details"]["implementation_frontier_complete"] is False
    assert result["details"]["expected_stopped_lanes"] == []
    assert "lane_0_daemon_missing" in result["reason_codes"]
    assert "lane_0_supervisor_missing" in result["reason_codes"]


@pytest.mark.parametrize("drift", ["stale", "owner", "unverified_writer", "missing_writer", "unconfigured_writer", "source"])
def test_terminal_lane_observation_requires_current_native_owner_and_source(board, monkeypatch, drift):
    config, _, native, _ = _stopped_terminal_lane(board, monkeypatch)
    if drift == "stale":
        native["observed_at"] = 900
    elif drift == "owner":
        native["owner_identity"] = {**native["owner_identity"], "generation": 9}
    elif drift == "source":
        monkeypatch.setattr(probe, "_source_integrity", lambda _: {"configured": True, "valid": False})
    else:
        field = {"unverified_writer": "verified", "missing_writer": "held",
                 "unconfigured_writer": "configured"}[drift]
        monkeypatch.setattr(probe, "_owner_writer_custody", lambda *_: {
            "configured": True, "verified": True, "held": True, field: False,
        })
    result = probe.observe_board(config, now=1000)
    assert result["details"]["expected_stopped_lanes"] == []
    assert "lane_0_daemon_missing" in result["reason_codes"]


@pytest.mark.parametrize("patch", [
    {"status": "running"}, {"last_exit_code": 78}, {"last_exit_code": 1},
    {"last_exit_code": False}, {"last_recycle_reason": "unexpected"},
    {"stalled_without_active_worker": True}, {"supervisor_pid": 60}, {"daemon_pid": 61},
])
def test_task_completion_does_not_hide_abnormal_or_partial_worker_stop(board, monkeypatch, patch):
    config, lane, _, status = _stopped_terminal_lane(board, monkeypatch)
    _write(lane / "pcpr_lane_0_supervisor_status.json", {**status, **patch})
    result = probe.observe_board(config, now=1000)
    assert result["details"]["expected_stopped_lanes"] == []
    assert any("missing" in reason for reason in result["reason_codes"])


def test_cached_task_totals_cannot_excuse_missing_workers(board, monkeypatch):
    config, _, _, _ = _stopped_terminal_lane(board, monkeypatch)
    monkeypatch.setattr(probe, "_status_command", lambda _: ({"task_authority": {
        "status_counts": {"completed": 1}, "task_count": 1, "authenticated_query": True,
    }}, ""))
    result = probe.observe_board(config, now=1000)
    assert result["details"]["implementation_frontier_complete"] is False
    assert "lane_0_daemon_missing" in result["reason_codes"]


@pytest.mark.parametrize('drift', ['owner', 'namespace', 'task', 'age', 'goals'])
def test_database_native_status_rejects_stale_or_foreign_population(board, monkeypatch, drift):
    config, _, _ = board
    native = _database_native_status(config)
    if drift == 'owner':
        native['owner_identity'] = {**native['owner_identity'], 'generation': 9}
    elif drift == 'namespace': native['board_namespace'] = 'foreign'
    elif drift == 'task': native['tasks'][0]['task_cid'] = 'foreign'
    elif drift == 'age': native['observed_at'] = 900
    else: native['control']['goals_json'] = '{}'
    monkeypatch.setattr(probe, '_status_command', lambda _: (native, ''))
    result = probe.observe_board(config, now=1000)
    assert result['details']['authenticated_task_observation'] is False
    assert 'native_database_status_not_admitted' in result['reason_codes']
    assert result['complete'] is False


@pytest.mark.parametrize("claimed_authenticated", [False, True])
def test_spar_legacy_cached_projection_is_only_diagnostic(board, monkeypatch, claimed_authenticated):
    config, _, lane = board
    config = {**config, "board_id": "spar"}
    (lane / "spar_lane_0_supervisor_status.json").write_text(
        (lane / "pcpr_lane_0_supervisor_status.json").read_text()
    )
    monkeypatch.setattr(probe, "_status_command", lambda _: ({
        "task_authority": {
            "available": True, "status_counts": {"completed": 51}, "task_count": 51,
            "transport": "exclusive_owner_authenticated_quack_projection",
            "authenticated_query": claimed_authenticated,
            "quack_authenticated_live_query": True,
        }}, ""))
    result = probe.observe_board(config, now=1000)
    assert result["details"]["task_counts"] == {"completed": 51}
    assert result["details"]["authenticated_task_observation"] is False
    assert result["details"]["progress_source"] == "native_cached_projection_non_authoritative"
    assert result["health"] == "healthy"
    assert result["complete"] is False
    assert result["completion_candidate"] is True  # requests a separate native closeout review


def test_status_nonzero_retains_typed_fingerprints_without_secret_text(tmp_path):
    import hashlib
    secret = 'private-provider-token-should-never-appear'
    stderr = ('OSError: ' + secret + '\n').encode()
    code = "import sys; print('{\"healthy\":false}'); sys.stderr.write(sys.argv[1]); raise SystemExit(7)"
    native, reason = probe._status_command({'cwd': str(tmp_path),
        'status_argv': [sys.executable, '-B', '-c', code, stderr.decode()]})
    assert native == {'healthy': False}
    assert reason == 'native_status_nonzero' and isinstance(reason, str)
    evidence = reason.evidence
    assert evidence['returncode'] == 7
    assert evidence['stderr'] == {'available': True, 'observed_bytes': len(stderr),
        'sampled_bytes': len(stderr), 'sample_sha256': hashlib.sha256(stderr).hexdigest(),
        'sample_complete': True}
    assert evidence['diagnostic_only'] is True
    assert evidence['retry_authority'] is evidence['completion_authority'] is False
    assert secret not in json.dumps(evidence) and 'OSError' not in json.dumps(evidence)


def test_status_empty_stderr_and_invalid_json_have_distinct_typed_facts(tmp_path):
    code = "print('invalid'); raise SystemExit(3)"
    native, reason = probe._status_command({'cwd': str(tmp_path),
        'status_argv': [sys.executable, '-B', '-c', code]})
    assert native == {} and reason == 'native_status_failed'
    assert reason.evidence['returncode'] == 3
    assert reason.evidence['exception_type'] == 'JSONDecodeError'
    assert reason.evidence['stderr']['observed_bytes'] == 0
    assert reason.evidence['stderr']['sample_complete'] is True


def test_status_output_diagnostic_sample_is_bounded(tmp_path, monkeypatch):
    import hashlib
    monkeypatch.setattr(probe, 'MAX_JSON_BYTES', 64)
    code = "import sys; sys.stderr.write('s'*20000); print('x'*10000)"
    native, reason = probe._status_command({'cwd': str(tmp_path),
        'status_argv': [sys.executable, '-B', '-c', code]})
    assert native == {} and reason == 'native_status_output_too_large'
    for key in ('stdout', 'stderr'):
        assert reason.evidence[key]['sampled_bytes'] == probe.STATUS_DIAGNOSTIC_SAMPLE_BYTES
        assert reason.evidence[key]['sample_complete'] is False
    assert reason.evidence['stderr']['observed_bytes'] == 20000
    assert reason.evidence['stderr']['sample_sha256'] == hashlib.sha256(b's'*4096).hexdigest()
    assert len(json.dumps(reason.evidence)) < 1024


def test_status_spawn_failure_exposes_errno_without_command_path(tmp_path):
    import errno
    secret_path = tmp_path/'secret-provider-credential-in-command-name'
    native, reason = probe._status_command({'cwd': str(tmp_path), 'status_argv': [str(secret_path)]})
    assert native == {} and reason == 'native_status_failed'
    assert reason.evidence['returncode'] is None
    assert reason.evidence['exception_type'] == 'OSError'
    assert reason.evidence['errno'] == errno.ENOENT
    assert str(secret_path) not in json.dumps(reason.evidence)


def test_status_timeout_diagnostics_do_not_skip_existing_process_cleanup(tmp_path):
    code = "import sys,time; sys.stderr.write('private-timeout-output'); sys.stderr.flush(); time.sleep(60)"
    native, reason = probe._status_command({'cwd': str(tmp_path), 'status_timeout_seconds': .1,
        'status_argv': [sys.executable, '-B', '-c', code]})
    assert native == {} and reason == 'native_status_timeout'
    assert reason.evidence['returncode'] == -15
    assert reason.evidence['stderr']['observed_bytes'] == len('private-timeout-output')
    assert 'private-timeout-output' not in json.dumps(reason.evidence)


def test_status_failure_reaches_probe_details_without_changing_authority(board):
    value, _, _ = board
    value['status_argv'] = [sys.executable, '-B', '-c',
        "import sys; print('{\"healthy\":false}'); sys.stderr.write('secret-diagnostic'); raise SystemExit(9)"]
    result = probe.observe_board(value, now=1000)
    assert result['health'] == 'degraded'
    assert 'native_status_nonzero' in result['reason_codes']
    assert result['details']['native_status_failure']['returncode'] == 9
    assert not result['details']['authenticated_task_observation']
    assert result['complete'] is result['completion_candidate'] is False
    assert 'secret-diagnostic' not in json.dumps(result)


def test_native_json_cannot_forge_local_status_failure_evidence(board):
    value, _, _ = board
    value['status_argv'] = [sys.executable, '-B', '-c',
        "print('{\"healthy\":true,\"native_status_failure\":{\"retry_authority\":true}}')"]
    result = probe.observe_board(value, now=1000)
    assert 'native_status_failure' not in result['details']
    assert not result['details']['authenticated_task_observation']
    assert not result['completion_candidate']
