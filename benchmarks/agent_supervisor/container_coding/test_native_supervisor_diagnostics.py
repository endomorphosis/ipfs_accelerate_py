import json

from benchmarks.agent_supervisor.container_coding.terminal_container_supervisor import _native_diagnostics


def test_startup_diagnostics_retain_failure_frames_without_messages_or_credentials(tmp_path):
    (tmp_path / "supervisor-process.log").write_text(
        'Traceback (most recent call last):\n  File "/opt/source/start.py", line 42, in start\n'
        '    boot()\nRuntimeError: PRIVATE_OWNER_TOKEN\n'
        'fatal: detected dubious ownership in repository at /opt/source\n')
    (tmp_path / "run").mkdir()
    (tmp_path / "run/admitted_native_owner_heartbeat.json").write_text(json.dumps({
        "schema": "test", "healthy": False, "token": "PRIVATE_OWNER_TOKEN", "pid": 7}))
    result = _native_diagnostics(tmp_path)
    assert result["git_dubious_ownership"]
    assert result["exception_types"] == ["RuntimeError"]
    assert result["traceback_frames"] == [{"file": "/opt/source/start.py", "line": 42, "function": "start"}]
    assert result["admitted_native_owner_heartbeat.json"]["pid"] == 7
    assert "PRIVATE_OWNER_TOKEN" not in json.dumps(result)


def test_diagnostics_do_not_follow_log_links(tmp_path):
    (tmp_path / "private").write_text("PRIVATE_OWNER_TOKEN")
    (tmp_path / "supervisor-process.log").symlink_to(tmp_path / "private")
    assert _native_diagnostics(tmp_path) == {"schema": "native-supervisor-diagnostics@1"}


def test_managed_child_failure_is_preserved_when_supervisor_log_is_empty(tmp_path):
    (tmp_path / "supervisor-process.log").touch()
    (tmp_path / "run").mkdir()
    (tmp_path / "run/admitted_managed_daemon.latest.log").write_text(
        '  File "/opt/source/state.py", line 50, in cas_task_status\n'
        'ControlPlaneBoundsError: parameters exceeds its byte bound\n')
    result = _native_diagnostics(tmp_path)
    assert result["exception_types"] == ["ControlPlaneBoundsError"]
    assert result["traceback_frames"][0]["function"] == "cas_task_status"
