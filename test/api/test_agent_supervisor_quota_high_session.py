"""Quota-high command binding and native workspace-scoped session admission."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
from urllib.parse import quote

import pytest

from ipfs_accelerate_py import agent_implementation_route as route
from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon as daemon_module


@pytest.fixture
def pinned_daemon(tmp_path, monkeypatch):
    for key in list(os.environ):
        if key.startswith("IPFS_ACCELERATE_AGENT_IMPLEMENTATION_ROUTE_"):
            monkeypatch.delenv(key)
    monkeypatch.delenv("IMPLEMENTATION_DAEMON_COMMAND", raising=False)
    monkeypatch.setenv(daemon_module.IMPLEMENTATION_PROVIDER_ENV, "grok")
    workspace = tmp_path / "worktrees" / "worker"
    workspace.mkdir(parents=True)
    board = tmp_path / "tasks.todo.md"
    board.write_text("# Agent Todos\n")
    daemon = daemon_module.TodoImplementationDaemon(
        todo_path=board, state_path=tmp_path / "state.json",
        strategy_path=tmp_path / "strategy.json", events_path=tmp_path / "events.jsonl",
        repo_root=tmp_path, worktree_root=workspace.parent)
    task = daemon_module.PortalTask(task_id="AF-002", title="Freeze experiment protocol",
        status="ready", completion="auto", priority="P0", track="autoformalization",
        outputs=["papers/completion/autoformalization/protocol.md"],
        validation=["python3 scripts/paper_supervisors.py verify-task --paper autoformalization --task AF-002"],
        acceptance="Record executable honest protocol.")
    real_popen = subprocess.Popen
    def prohibit_provider_process(argv, *args, **kwargs):
        if isinstance(argv, (tuple, list)) and argv:
            assert Path(str(argv[0])).name not in {"grok", "codex", "claude", "copilot"}
        return real_popen(argv, *args, **kwargs)
    monkeypatch.setattr(subprocess, "Popen", prohibit_provider_process)
    return daemon, workspace, task


def test_actual_quota_high_route_keeps_high_and_fresh_nonce(pinned_daemon, monkeypatch):
    daemon, workspace, task = pinned_daemon
    if not daemon_module._grok_cli_available() or not shutil.which("codex"):
        pytest.skip("Installed authenticated Grok and Codex binary required")
    for key, value in {
        daemon_module.IMPLEMENTATION_FALLBACK_PROVIDER_ENV: "codex",
        daemon_module.IMPLEMENTATION_FALLBACK_TRIGGER_ENV: "primary_quota_exhausted",
        daemon_module._GROK_MODEL_ENV: "grok-4.6",
        daemon_module._CODEX_MODEL_ENV: "gpt-5.6-terra",
        daemon_module._CODEX_REASONING_EFFORT_ENV: "high",
    }.items():
        monkeypatch.setenv(key, value)
    commands = [daemon._build_implementation_command(workspace, task=task) for _ in range(2)]
    nonces = []
    for command in commands:
        assert "--canonical-legacy-preflight-route" not in command
        binding = json.loads(command[command.index("--agent-implementation-route-json") + 1])
        parsed = route.resolve_agent_implementation_route_binding(binding, repo_root=workspace)
        assert parsed.fallback_reasoning_effort == "high"
        assert parsed.fallback_trigger == "primary_quota_exhausted"
        assert parsed.permits_authentication_unavailable is False
        fallback = json.loads(command[command.index("--codex-fallback-command-json") + 1])
        assert fallback[fallback.index("-m") + 1] == "gpt-5.6-terra"
        assert 'model_reasoning_effort="high"' in fallback
        nonce = command[command.index("--grok-failure-receipt-nonce") + 1]
        assert len(bytes.fromhex(nonce)) == 32
        nonces.append(nonce)
    assert nonces[0] != nonces[1]


@pytest.fixture
def native_session(tmp_path):
    workspace = tmp_path / "workspace !'()* é"
    workspace.mkdir()
    home = tmp_path / "grok-home"
    session_id = "f159e13e-462f-43bc-9da2-01bd0c1f5761"
    session = home / "sessions" / quote(str(workspace), safe="!'()*-._~") / session_id
    session.mkdir(parents=True)
    message = route._AGENT_IMPLEMENTATION_SPENDING_LIMIT_MESSAGE
    updates = [
        {"sessionUpdate": "retry_state", "type": "failed", "error_type": "api", "message": message},
        {"sessionUpdate": "turn_completed", "stop_reason": "error", "agent_result": message},
    ]
    (session / "updates.jsonl").write_text("".join(json.dumps({"method": "session/update", "params": {
        "sessionId": session_id, "update": update}}) + "\n" for update in updates))
    (session / "summary.json").write_text(json.dumps({"info": {"id": session_id, "cwd": str(workspace)},
        "current_model_id": "grok-4.6", "grok_home": str(home)}))
    for path in session.iterdir():
        path.chmod(0o600)
    receipt = route.build_agent_implementation_failure_receipt(
        probe_stderr_text="Grok Build usage balance exhausted", nonce="a" * 64,
        model="grok-4.6", probe_returncode=1)
    args = {"grok_home": home, "expected_session_id": session_id, "verifier_returncode": 1,
            "failure_receipt": receipt, "verifier_workspace": workspace}
    return args, session


def test_exact_encoded_workspace_native_quota_admitted(native_session):
    args, _ = native_session
    evidence = route.validate_agent_implementation_quota_evidence(**args)
    assert evidence is not None
    assert evidence.verifier_result == "spending_limit_exhausted"


@pytest.mark.parametrize("mutation", ["different_workspace", "wrong_session", "summary_cwd",
    "wrong_model", "no_terminal", "ambiguous", "symlink", "fifo"])
def test_native_quota_rejects_inexact_or_ambiguous_evidence(native_session, tmp_path, mutation):
    args, session = native_session
    if mutation == "different_workspace":
        other = tmp_path / "other"
        other.mkdir()
        args["verifier_workspace"] = other
    elif mutation == "wrong_session":
        args["expected_session_id"] = "815514d0-5c19-4e17-aec9-2ea8b920edb8"
    elif mutation in {"summary_cwd", "wrong_model"}:
        path = session / "summary.json"
        value = json.loads(path.read_text())
        if mutation == "summary_cwd":
            value["info"]["cwd"] = str(tmp_path)
        else:
            value["current_model_id"] = "other-model"
        path.write_text(json.dumps(value))
    elif mutation == "no_terminal":
        path = session / "updates.jsonl"
        path.write_text(path.read_text().splitlines()[0] + "\n")
    elif mutation == "ambiguous":
        shutil.copytree(session, args["grok_home"] / "sessions" / args["expected_session_id"])
    elif mutation == "symlink":
        moved = session.with_name("moved")
        session.rename(moved)
        session.symlink_to(moved, target_is_directory=True)
    else:
        path = session / "updates.jsonl"
        path.unlink()
        os.mkfifo(path, mode=0o600)
    assert route.validate_agent_implementation_quota_evidence(**args) is None


def test_quota_session_directory_swap_is_rejected(native_session, monkeypatch):
    args, session = native_session
    real_read = route._read_stable_agent_implementation_evidence_file
    swapped = False
    def read_then_swap(path, **kwargs):
        nonlocal swapped
        value = real_read(path, **kwargs)
        if Path(path).name == "summary.json" and not swapped:
            swapped = True
            old = session.with_name("old")
            session.rename(old)
            shutil.copytree(old, session)
        return value
    monkeypatch.setattr(route, "_read_stable_agent_implementation_evidence_file", read_then_swap)
    assert route.validate_agent_implementation_quota_evidence(**args) is None
    assert swapped
