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
