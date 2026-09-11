"""Native quota evidence must follow its exact independent verifier workspace."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import quote

import pytest

from ipfs_accelerate_py import agent_implementation_route as route
from ipfs_accelerate_py.agent_supervisor.runtime import grok_cli_runner as runner


SESSION = "00000000-0000-4000-8000-000000000001"
MESSAGE = "API error (status 402 Payment Required): Grok Build usage balance exhausted"


def receipt():
    return runner.build_grok_failure_receipt(
        probe_stderr_text=MESSAGE, nonce="a" * 64, model="grok-4.6",
        probe_returncode=1, primary_dispatched=False,
    )


def write_session(home, workspace=None, session_id=SESSION):
    path = home / "sessions"
    if workspace is not None:
        path /= quote(str(workspace), safe="!'()*-._~")
    path /= session_id
    path.mkdir(parents=True, mode=0o700)
    updates = [
        {"sessionUpdate": "retry_state", "type": "failed", "error_type": "api", "message": MESSAGE},
        {"sessionUpdate": "turn_completed", "stop_reason": "error", "agent_result": MESSAGE},
    ]
    (path / "updates.jsonl").write_text("".join(json.dumps({
        "method": "_x.ai/session/update", "params": {
            "sessionId": session_id, "update": update,
        },
    }) + "\n" for update in updates))
    (path / "summary.json").write_text(json.dumps({
        "info": {"id": session_id, **({"cwd": str(workspace)} if workspace else {})},
        "current_model_id": "grok-4.6", "grok_home": str(home),
    }))
    return path


def validate(home, workspace=None, failure_receipt=None):
    return route.validate_agent_implementation_quota_evidence(
        grok_home=home, expected_session_id=SESSION, verifier_returncode=1,
        failure_receipt=failure_receipt if failure_receipt is not None else receipt(), verifier_workspace=workspace,
    )


@pytest.mark.parametrize("layout", ["legacy", "workspace", "punctuation"])
def test_valid_native_quota_layout_returns_bound_evidence(tmp_path, layout):
    home = tmp_path / "home"
    workspace = None if layout == "legacy" else tmp_path / (
        "workspace-!'()*" if layout == "punctuation" else "workspace"
    )
    if workspace is not None:
        workspace.mkdir()
    write_session(home, workspace)
    failure_receipt = receipt()
    evidence = validate(home, workspace, failure_receipt)
    assert isinstance(evidence, route.AgentImplementationQuotaEvidence)
    decision = route.decide_agent_implementation_fallback(
        route.resolve_agent_implementation_route(
            primary_provider_id="grok_cli", primary_model_id="grok-4.6",
            fallback_provider_id="codex", fallback_model_id="gpt-5.6-terra",
            fallback_trigger="primary_quota_exhausted", fallback_reasoning_effort="high",
        ),
        repo_root=tmp_path, failure_receipt=failure_receipt, expected_nonce="a" * 64,
        expected_model="grok-4.6", expected_probe_returncode=1,
        independent_quota_evidence=evidence,
    )
    assert decision.authorized is True


@pytest.mark.parametrize("mutation", [
    "legacy_duplicate", "legacy_only", "foreign_workspace", "summary_cwd",
    "summary_model", "terminal_mismatch", "authentication", "rate_limit",
    "extra_event", "session_symlink", "workspace_symlink", "transcript_fifo",
])
def test_native_quota_layout_refuses_ambiguity_and_foreign_evidence(tmp_path, mutation):
    home, workspace = tmp_path / "home", tmp_path / "workspace"
    workspace.mkdir()
    session = write_session(home, workspace)
    if mutation in {"legacy_duplicate", "legacy_only"}:
        write_session(home)
        if mutation == "legacy_only":
            session.rename(session.with_name("unselected"))
    elif mutation == "foreign_workspace":
        workspace = tmp_path / "another-workspace"
        workspace.mkdir()
    elif mutation in {"summary_cwd", "summary_model"}:
        summary = json.loads((session / "summary.json").read_text())
        if mutation == "summary_cwd":
            summary["info"]["cwd"] = str(tmp_path)
        else:
            summary["current_model_id"] = "grok-4.5"
        (session / "summary.json").write_text(json.dumps(summary))
    elif mutation in {"terminal_mismatch", "authentication", "rate_limit", "extra_event"}:
        events = [json.loads(line) for line in (session / "updates.jsonl").read_text().splitlines()]
        if mutation == "terminal_mismatch":
            events[-1]["params"]["update"]["agent_result"] = "different failure"
        elif mutation == "extra_event":
            events.append({"method": "session/update", "params": {
                "sessionId": SESSION, "update": {"sessionUpdate": "tool_call"},
            }})
        else:
            text = "Not signed in" if mutation == "authentication" else "HTTP 429 Too Many Requests"
            events[0]["params"]["update"]["message"] = text
            events[-1]["params"]["update"]["agent_result"] = text
        (session / "updates.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    elif mutation == "session_symlink":
        moved = session.with_name("moved")
        session.rename(moved)
        session.symlink_to(moved, target_is_directory=True)
    elif mutation == "workspace_symlink":
        linked = tmp_path / "linked-workspace"
        linked.symlink_to(workspace, target_is_directory=True)
        workspace = linked
    else:
        (session / "updates.jsonl").unlink()
        os.mkfifo(session / "updates.jsonl")
    assert validate(home, workspace) is None


def test_native_quota_session_directory_replacement_is_refused(tmp_path, monkeypatch):
    home, workspace = tmp_path / "home", tmp_path / "workspace"
    workspace.mkdir()
    session = write_session(home, workspace)
    original = route._read_stable_agent_implementation_evidence_file

    def swapped(path, **kwargs):
        result = original(path, **kwargs)
        if path.name == "summary.json":
            session.rename(session.with_name("retained"))
            write_session(home, workspace)
        return result

    monkeypatch.setattr(route, "_read_stable_agent_implementation_evidence_file", swapped)
    assert validate(home, workspace) is None


def test_native_quota_file_fifo_replacement_does_not_block(tmp_path):
    import multiprocessing
    path = tmp_path / "evidence.json"
    path.write_text("{}")
    path.chmod(0o600)

    def check():
        original_open = os.open

        def replace_file(value, flags, *args, **kwargs):
            if Path(value) == path:
                path.unlink()
                os.mkfifo(path)
            return original_open(value, flags, *args, **kwargs)

        os.open = replace_file
        assert route._read_stable_agent_implementation_evidence_file(path) is None

    child = multiprocessing.get_context("fork").Process(target=check)
    child.start()
    try:
        child.join(2)
        assert not child.is_alive(), "native evidence FIFO blocked the verifier"
        assert child.exitcode == 0
    finally:
        if child.is_alive():
            child.terminate()
        child.join(2)
        child.close()


def test_independent_verifier_uses_short_cwd_and_native_namespaced_session(tmp_path, monkeypatch):
    home = tmp_path / "private-grok-home"
    deep = tmp_path.joinpath(*(["inherited-temp"] * 25))
    deep.mkdir(parents=True)
    monkeypatch.setattr(runner.tempfile, "tempdir", str(deep))
    captured = {}

    def isolated(**_kwargs):
        home.mkdir(mode=0o700)
        return SimpleNamespace(name=str(home), cleanup=lambda: None), {
            "GROK_HOME": str(home), "OLDPWD": "/task/worktree",
        }, home / "settings.json", ()

    def run(command, **kwargs):
        workspace = kwargs["cwd"]
        captured.update(cwd=workspace, command=command, kwargs=kwargs)
        session_id = command[command.index("--session-id") + 1]
        write_session(home, workspace, session_id)
        return subprocess.CompletedProcess(command, 1)

    monkeypatch.setattr(runner, "_isolated_grok_home", isolated)
    monkeypatch.setattr(runner.subprocess, "run", run)
    evidence = runner._independently_verify_grok_quota(
        grok_bin="/opt/providers/grok", base_env={}, failure_receipt=receipt(),
    )
    assert isinstance(evidence, route.AgentImplementationQuotaEvidence)
    assert captured["cwd"].parent.parent == Path("/tmp")
    assert captured["kwargs"]["umask"] == 0o077
    assert captured["kwargs"]["env"]["PWD"] == str(captured["cwd"])
    assert "OLDPWD" not in captured["kwargs"]["env"]
    command = captured["command"]
    assert command[command.index("--tools") + 1] == ""
    assert command[command.index("--max-turns") + 1] == "1"
    assert not captured["cwd"].exists()
    assert not home.exists()
