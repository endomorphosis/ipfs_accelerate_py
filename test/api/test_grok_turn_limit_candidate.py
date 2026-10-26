"""Strict terminal observations and native validation/train after a turn limit.

All model and Docker boundaries are explicit fixtures; local child processes,
Git candidates, proposal validation, declared checks and queue state are real.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

from ipfs_accelerate_py.agent_supervisor.validation import grok_turn_limit as guard
from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon as native
from test.api.test_provider_failure_checkpoint_retention import failed_child_case, git, run_failed_child


def command(workspace):
    return [sys.executable, "-m", guard._MODULE, "--workspace", str(workspace),
            "--model", "grok-4.7", "--max-turns", "2", "--mode", "agent"]


def stream(workspace):
    return [json.dumps({"type": "tool_call_update", "rawOutput": "fixture XAI 402 quota error"}),
            json.dumps({"type": "max_turns_reached"}),
            json.dumps({"type": "end", "stopReason": "max_turn_requests", "num_turns": 2, "cwd": str(workspace)}),
            "provider command available: python", "Error: max turns reached"]


def log_case(tmp_path, lines=None):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    argv = command(workspace)
    prefix = (f"Task: TEST-1 Synthetic source\nStarted: fixture-start\nWorkspace: {workspace}\n"
              f"Branch: fixture\nBaseline: {'a' * 40}\nCommand: {shlex.join(argv)}\n\n")
    log = tmp_path / "provider.log"
    log.write_text(prefix + "\n".join(stream(workspace) if lines is None else lines) + "\n")
    log.chmod(0o600)
    options = dict(command=argv, workspace=workspace, task_id="TEST-1", title="Synthetic source",
                   started_at="fixture-start", branch="fixture", baseline_ref="a" * 40)
    return log, options


def test_real_terminal_shape_and_opaque_tool_quota_text_admit_only_validation(tmp_path):
    log, options = log_case(tmp_path)
    observed = guard.observe_terminal(log, **options)
    assert observed["provider_returncode"] == 1 and observed["num_turns"] == 2
    assert observed["terminal_reason"] == "max_turn_requests"
    assert all(observed[key] is False for key in ("completion_authority", "attempt_refunded", "new_provider_authority"))
    assert "quota error" not in json.dumps(observed)


@pytest.mark.parametrize("damage", ["partial", "duplicate_end", "duplicate_turn", "post_end_json",
    "typed_error", "early_quota", "unknown_frame", "boolean_turns", "wrong_turns", "wrong_cwd",
    "duplicate_keys", "malformed", "plain_error", "missing_max", "missing_end", "missing_stderr",
    "wrong_stop", "preamble", "foreign_command", "fallback", "protected", "oversized"])
def test_incomplete_error_unbound_or_ambiguous_terminal_never_admits(tmp_path, monkeypatch, damage):
    log, options = log_case(tmp_path)
    raw = log.read_text()
    if damage == "partial": raw = raw[:-1]
    elif damage == "duplicate_end": raw += json.dumps({"type": "end"}) + "\n"
    elif damage == "duplicate_turn": raw = raw.replace('{"type": "max_turns_reached"}', '{"type": "max_turns_reached"}\n{"type": "max_turns_reached"}')
    elif damage == "post_end_json": raw += '{"type":"usage"}\n'
    elif damage in {"typed_error", "early_quota", "unknown_frame"}:
        frame = {"type": "unknown_frame"} if damage == "unknown_frame" else {"type": "error", "message": "402 quota exhausted"}
        raw = raw.replace('{"type": "tool_call_update"', json.dumps(frame) + '\n{"type": "tool_call_update"', 1)
        if damage == "early_quota": raw = raw.replace('fixture XAI 402 quota error', 'x' * 100_000)
    elif damage == "boolean_turns": raw = raw.replace('"num_turns": 2', '"num_turns": true')
    elif damage == "wrong_turns": raw = raw.replace('"num_turns": 2', '"num_turns": 3')
    elif damage == "wrong_cwd": raw = raw.replace('"cwd": "' + str(options["workspace"]), '"cwd": "/foreign')
    elif damage == "duplicate_keys": raw = raw.replace('"stopReason": "max_turn_requests"', '"stopReason":"other","stopReason":"max_turn_requests"')
    elif damage == "malformed": raw += '{broken}\n'
    elif damage == "plain_error": raw += 'Error: authentication failed\n'
    elif damage == "missing_max": raw = raw.replace('{"type": "max_turns_reached"}\n', '')
    elif damage == "missing_end": raw = '\n'.join(line for line in raw.splitlines() if not line.startswith('{"type": "end"')) + '\n'
    elif damage == "missing_stderr": raw = raw.replace('Error: max turns reached\n', '')
    elif damage == "wrong_stop": raw = raw.replace('max_turn_requests', 'cancelled')
    elif damage == "preamble": raw = raw.replace('Task: TEST-1', 'Task: FOREIGN-1')
    elif damage == "foreign_command": options["command"] = [sys.executable, "-c", "pass", guard._MODULE]
    elif damage in {"fallback", "protected"}: options["command"] += ['--codex-fallback-command-json' if damage == "fallback" else '--agent-implementation-route-json', '{}']
    elif damage == "oversized": monkeypatch.setattr(guard, "MAX_LOG_BYTES", 12)
    log.write_text(raw)
    with pytest.raises((ValueError, UnicodeError)):
        guard.observe_terminal(log, **options)


def test_real_waited_process_and_fake_empty_docker_are_independent_quiescence_proofs(tmp_path, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime import grok_cli_runner as runtime
    argv = [sys.executable, "-c", "import time; time.sleep(.1); raise SystemExit(1)"]
    process = subprocess.Popen(argv, start_new_session=True)
    birth = native.PortalImplementationDaemon._provider_process_identity(process.pid)
    process.wait(timeout=5)
    monkeypatch.setattr(runtime, "_docker_isolation_binary", lambda: "/usr/bin/docker")
    calls = []
    def fake_docker(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, "", "")
    monkeypatch.setattr(guard.subprocess, "run", fake_docker)
    observed = guard.observe_quiescence(process, command=argv, birth=birth, workspace=tmp_path)
    assert observed["process_reaped"] and observed["process_group_absent"] and observed["workspace_containers_absent"]
    assert len(calls) == 1 and "ps" in calls[0]
    with pytest.raises(ValueError, match="custody"):
        guard.observe_quiescence(process, command=argv, birth=None, workspace=tmp_path)


@pytest.mark.parametrize("mode", ["owned_container", "foreign_container", "unknown_inventory", "over_bound"])
def test_docker_absence_is_observed_without_stopping_any_container(tmp_path, monkeypatch, mode):
    from ipfs_accelerate_py.agent_supervisor.runtime import grok_cli_runner as runtime
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    argv = [sys.executable, "-c", "import time; time.sleep(.1); raise SystemExit(1)"]
    process = subprocess.Popen(argv, start_new_session=True)
    birth = native.PortalImplementationDaemon._provider_process_identity(process.pid)
    process.wait(timeout=5)
    monkeypatch.setattr(runtime, "_docker_isolation_binary", lambda: "/usr/bin/docker")
    calls = []
    def docker(argv, **kwargs):
        calls.append(argv)
        assert "stop" not in argv and "rm" not in argv
        if mode == "unknown_inventory":
            raise subprocess.CalledProcessError(1, argv)
        if "ps" in argv:
            value = ("a" * 64 + "\n") * (33 if mode == "over_bound" else 1)
        else:
            value = json.dumps([{"Type": "bind", "Source": str(workspace if mode == "owned_container" else tmp_path / "other")}])
        return subprocess.CompletedProcess(argv, 0, value, "")
    monkeypatch.setattr(guard.subprocess, "run", docker)
    if mode == "foreign_container":
        assert guard.observe_quiescence(process, command=argv, birth=birth, workspace=workspace)["workspace_containers_absent"]
    else:
        with pytest.raises((ValueError, subprocess.SubprocessError)):
            guard.observe_quiescence(process, command=argv, birth=birth, workspace=workspace)
    assert calls


@pytest.fixture
def native_case(tmp_path, monkeypatch):
    daemon, _, state, _ = failed_child_case(tmp_path)
    daemon.task_header_prefix = "## RET-"
    daemon.todo_path.write_text("# Synthetic task\n\n## RET-001 Revise the declared value\n\n"
        "- Status: todo\n- Completion: validation\n- Priority: P1\n- Outputs: src/value.py\n"
        "- Acceptance: The declared value is revised and checked.\n"
        "- Validation: python -m pytest -q tests/test_value.py\n"
        "- Requires provider: true\n")
    (daemon.repo_root / "tests").mkdir()
    (daemon.repo_root / "tests/test_value.py").write_text(
        "from src.value import VALUE\n\ndef test_required_value():\n    assert VALUE == 2\n")
    git(daemon.repo_root, "add", "tasks.todo.md", "tests/test_value.py")
    git(daemon.repo_root, "commit", "-m", "Synthetic board")
    task = daemon._load_tasks()[0]
    monkeypatch.setattr(daemon, "_build_implementation_command", lambda workspace, **_: command(workspace))
    actual_stream = native.run_process_group_stream
    calls = []
    def synthetic_provider(argv, **kwargs):
        calls.append(argv)
        workspace = Path(kwargs["cwd"])
        child = tmp_path / "synthetic_grok.py"
        child.write_text("from pathlib import Path\nimport json\n"
            "Path('src/value.py').write_text('VALUE = 2\\n')\n"
            "print(json.dumps({'type':'max_turns_reached'}))\n"
            "print(json.dumps({'type':'end','stopReason':'max_turn_requests','num_turns':2,'cwd':str(Path.cwd())}))\n"
            "print('Error: max turns reached')\nraise SystemExit(1)\n")
        completed = actual_stream([sys.executable, str(child)], **kwargs)
        return subprocess.CompletedProcess(argv, completed.returncode)
    monkeypatch.setattr(native, "run_process_group_stream", synthetic_provider)
    # This explicit fake is the model/container boundary only. Separate tests
    # above exercise actual Popen/group proof and Docker refusal conditions.
    monkeypatch.setattr(guard, "observe_quiescence", lambda *a, **k:
        {"process_reaped": True, "process_group_absent": True, "workspace_containers_absent": True})
    monkeypatch.setattr(daemon, "_automatic_implementation_rescue", lambda **_:
        pytest.fail("turn-limit validation cannot dispatch or manufacture a rescue"))
    return daemon, task, state, calls


def test_turn_limit_candidate_runs_real_declared_validation_and_normal_train(native_case):
    daemon, task, state, calls = native_case
    result = run_failed_child(daemon, task, state)
    assert result["grok_turn_limit"]["eligible"], result
    assert result["validation_result"]["attempted"] and result["validation_result"]["passed"], result
    assert result["provider_returncode"] == result["grok_turn_limit"]["provider_returncode"] == 1
    assert result["attempt_consumed"] and state.implementation_attempts[task.task_id] == 1
    assert len(calls) == 1
    assert result["returncode"] == 0 and result["merge_result"].get("merged"), result
    assert result["board_completion"]["complete"] and state.last_implementation_returncode == 0
    assert daemon._load_tasks()[0].status == "completed"
    assert git(daemon.repo_root, "show", "HEAD:src/value.py") == "VALUE = 2"
    assert "- Status: completed" in daemon.todo_path.read_text()
    assert result["merge_result"]["todo_update_result"]["completion_receipts"][0]["status"] == "succeeded"
    events = [json.loads(line) for line in daemon.events_path.read_text().splitlines()]
    observation = next(event for event in events if event["type"] == "implementation_grok_turn_limit_observed")
    assert observation["provider_returncode"] == 1 and observation["completion_authority"] is False


def test_validated_handoff_waits_for_normal_merge_callback(native_case, monkeypatch):
    daemon, task, state, calls = native_case
    monkeypatch.setattr(daemon, "_consume_one_merge_candidate", lambda: None)
    result = run_failed_child(daemon, task, state)
    assert result["returncode"] == 0 and result["provider_returncode"] == 1
    assert result["validation_result"]["passed"] and result["merge_result"]["queued"]
    assert result["board_completion"]["pending_merge"] and not result["board_completion"]["complete"]
    assert daemon._load_tasks()[0].status == "todo" and len(calls) == 1
    assert daemon.merge_queue.pending_count() == 1
    assert state.implementation_attempts[task.task_id] == 1


def test_failed_declared_validation_keeps_spent_attempt_and_never_completes(native_case):
    daemon, task, state, calls = native_case
    check = daemon.repo_root / "tests/test_value.py"
    check.write_text(check.read_text().replace("VALUE == 2", "VALUE == 3"))
    git(daemon.repo_root, "add", "tests/test_value.py")
    git(daemon.repo_root, "commit", "-m", "Require a different synthetic value")
    result = run_failed_child(daemon, task, state)
    assert result["grok_turn_limit"]["eligible"], result
    assert result["validation_result"]["attempted"] and not result["validation_result"]["passed"], result
    assert result["returncode"] != 0 and result["provider_returncode"] == 1
    assert result["attempt_consumed"] and state.implementation_attempts[task.task_id] == 1
    assert len(calls) == 1 and not result["merge_result"].get("merged")
    assert task.task_id not in state.completed_task_ids and "- Status: todo" in daemon.todo_path.read_text()


@pytest.mark.parametrize("hold", ["manual", "waiting_user", "blocked_projection", "reserved", "protected", "strategy", "capacity_unknown"])
def test_held_or_unknown_candidate_is_preserved_without_validation_or_refund(native_case, monkeypatch, hold):
    daemon, task, state, calls = native_case
    if hold == "manual":
        daemon.todo_path.write_text(daemon.todo_path.read_text() + "- Manual hold: true\n")
    elif hold == "waiting_user":
        daemon.todo_path.write_text(daemon.todo_path.read_text().replace("- Status: todo", "- Status: waiting_user"))
    elif hold == "blocked_projection": state.blocked_task_ids = [task.task_id]
    elif hold == "reserved": state.external_reserved_task_ids = [task.task_id]
    elif hold == "protected": state.protected_implementation_attempts = {"synthetic-held": {"task_id": task.task_id}}
    elif hold == "strategy":
        daemon.strategy_path.parent.mkdir(parents=True, exist_ok=True)
        daemon.strategy_path.write_text(json.dumps({"blocked_tasks": [task.task_id]}))
    else:
        monkeypatch.setattr(daemon, "_retained_provider_capacity_deferral", lambda: {"custody_error": "unknown"})
    result = run_failed_child(daemon, task, state)
    assert not result["grok_turn_limit"]["eligible"], result
    assert not result["validation_result"]["attempted"]
    assert result["provider_returncode"] == 1 and result["attempt_consumed"]
    assert state.implementation_attempts[task.task_id] == 1 and len(calls) == 1
    assert task.task_id not in state.completed_task_ids
    assert not result["merge_result"].get("merged")


def test_protected_mutation_during_quiescence_never_becomes_a_fresh_baseline(native_case, monkeypatch):
    daemon, task, state, calls = native_case
    protected = daemon.repo_root / "protected.txt"
    protected.write_text("original protected content\n")
    git(daemon.repo_root, "add", "protected.txt")
    git(daemon.repo_root, "commit", "-m", "Protect synthetic owner content")
    daemon.implementation_protected_paths = ("protected.txt",)
    def mutate_during_observation(*args, workspace, **kwargs):
        (workspace / "protected.txt").write_text("unauthorized change\n")
        return {"process_reaped": True, "process_group_absent": True, "workspace_containers_absent": True}
    monkeypatch.setattr(guard, "observe_quiescence", mutate_during_observation)
    result = run_failed_child(daemon, task, state)
    assert result["returncode"] != 0 and not result["validation_result"]["attempted"]
    assert result.get("protected_path_violation"), result
    assert daemon._implementation_protected_incident_path().exists()
    assert task.task_id not in state.completed_task_ids and len(calls) == 1
    assert protected.read_text() == "original protected content\n"


def test_operator_hold_written_during_quiescence_survives_attempt_finalization(native_case, monkeypatch):
    daemon, task, state, calls = native_case
    def hold_during_observation(*args, **kwargs):
        durable = native.PortalTaskState.load(daemon.state_path)
        durable.blocked_task_ids.append(task.task_id)
        durable.task_statuses[task.task_id] = "manual_hold"
        durable.save(daemon.state_path)
        return {"process_reaped": True, "process_group_absent": True, "workspace_containers_absent": True}
    monkeypatch.setattr(guard, "observe_quiescence", hold_during_observation)
    result = run_failed_child(daemon, task, state)
    assert not result["grok_turn_limit"]["eligible"] and not result["validation_result"]["attempted"]
    after = native.PortalTaskState.load(daemon.state_path)
    assert task.task_id in after.blocked_task_ids and after.task_statuses[task.task_id] == "manual_hold"
    assert after.implementation_attempts[task.task_id] == 1 and len(calls) == 1
    assert daemon._load_tasks()[0].status == "todo"
